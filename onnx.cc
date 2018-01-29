#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include "onnx.pb.h"
#include "thnets.h"

using namespace std;
using namespace google::protobuf::io;

static const onnx::TensorProto *getinitializer(const onnx::GraphProto *graph, const string &name)
{
	for(int i = 0; i < graph->initializer_size(); i++)
		if(name == graph->initializer(i).name())
			return &graph->initializer(i);
	// Not found in initializers, see if it's calculated
	for (int i = 0; i < graph->node_size(); i++)
		if(graph->node(i).output(0) == name && graph->node(i).input_size() == 1)
			return getinitializer(graph, graph->node(i).input(0));
	return 0;
}

extern "C" THFloatTensor *onnx_gettensor(const void *graph, int nodeidx, int inputidx)
{
	const onnx::GraphProto *g = (const onnx::GraphProto *)graph;
	if(inputidx >= g->node(nodeidx).input_size())
		return THFloatTensor_new();
	const onnx::TensorProto *t = getinitializer(g, g->node(nodeidx).input(inputidx));
	if(t->data_type() != 1)
		THError("Only float tensors are supported, got data_type %d for %s\n", t->data_type(), t->name().c_str());
	THFloatTensor *t1 = THFloatTensor_new();
	long sizes[4], total = 1;
	for(int i = 0; i < t->dims_size(); i++)
	{
		sizes[i] = t->dims(i);
		total *= sizes[i];
	}
	THFloatTensor_resize(t1, sizes, t->dims_size());
	memcpy(t1->storage->data, t->raw_data().c_str(), total * sizeof(float));
	return t1;
}

extern "C" int onnx_getint(const void *graph, int nodeidx, const char *attrname, int idx)
{
	const onnx::GraphProto *g = (const onnx::GraphProto *)graph;
	for(int i = 0; i < g->node(nodeidx).attribute_size(); i++)
	{
		const onnx::AttributeProto &attr = g->node(nodeidx).attribute(i);
		if(!strcmp(attr.name().c_str(), attrname))
		{
			if(idx == -1)
				return attr.i();
			if(idx < attr.ints_size())
				return attr.ints(idx);
			return 0;
		}
	}
	return 0;
}

extern "C" float onnx_getfloat(const void *graph, int nodeidx, const char *attrname, int idx)
{
	const onnx::GraphProto *g = (const onnx::GraphProto *)graph;
	for(int i = 0; i < g->node(nodeidx).attribute_size(); i++)
	{
		const onnx::AttributeProto &attr = g->node(nodeidx).attribute(i);
		if(!strcmp(attr.name().c_str(), attrname))
		{
			if(idx == -1)
				return attr.f();
			if(idx < attr.floats_size())
				return attr.floats(idx);
			return 0;
		}
	}
	return 0;
}

static struct {
	const char *name;
	void (*onnxload)(const void *, module *m, int nodeidx);
} name2loadf[] =
{
	{"Conv", onnxload_SpatialConvolution},
	{"ConvTranspose", onnxload_SpatialConvolutionTransposed},
	{"Gemm", onnxload_Linear},
	{"BatchNormalization", onnxload_SpatialBatchNormalization},
	{"MaxPool", onnxload_SpatialMaxPooling},
	{"Relu", onnxload_Threshold},
	{"Dropout", onnxload_Dropout},
	{"Softmax", onnxload_SoftMax},
	{"LogSoftmax", onnxload_LogSoftMax},
	{"Reshape", onnxload_View},
	{"Flatten", onnxload_View},
	{"Sum", onnxload_Add},
	{"Add", onnxload_Add},
	{"AveragePool", onnxload_SpatialAveragePooling},
	{"GlobalAveragePool", onnxload_SpatialAveragePooling},
	{"Concat", onnxload_Concat},
	{"Max", onnxload_Cmax},
	{"Slice", onnxload_Slice}
};

static int getfunction(const char *name)
{
	for(unsigned j = 0; j < sizeof(name2loadf)/sizeof(*name2loadf); j++)
		if(!strcmp(name, name2loadf[j].name))
			return j;
	return -1;
}

static int getoutput(struct network *net, const char *name)
{
	for(int i = 0; i < net->nelem; i++)
		if(!strcmp(net->modules[i].outputname, name))
			return i;
	return -1;
}

extern "C" struct network *loadonnx(const char* modelpath)
{
	GOOGLE_PROTOBUF_VERIFY_VERSION;
	int j;

	// Read the model protobuf
	int f = open(modelpath, O_RDONLY);
	if(f == -1)
		return 0;
	FileInputStream input(f);
	CodedInputStream cinput(&input);
	cinput.SetTotalBytesLimit(1024*1024*1024, -1);
	onnx::ModelProto model;
	if(!model.ParseFromCodedStream(&cinput))
	{
		close(f);
		return 0;
	}
	close(f);

	const onnx::GraphProto& graph=model.graph();

	// Build thnets::network object from onnx::Graph
	network *net = (network *)malloc(sizeof(*net));
	net->engine = ENGINE_CPU;
	// Overallocate modules by a factor of 2, because of split
	net->modules = (module *)calloc(graph.node_size() * 2, sizeof(*net->modules));
	net->nelem = 0;
	int n = 0;
	for (int i = 0; i < graph.node_size(); i++)
	{
		const onnx::NodeProto& node = graph.node(i);
		if(th_debug > 1)
		{
			printf("%s (%s): ", node.op_type().c_str(), node.name().c_str());
			for(int j = 0; j < node.input_size(); j++)
				if(!getinitializer(&graph, node.input(j)))
					printf("%s ", node.input(j).c_str());
			printf("-> ");
			for(int j = 0; j < node.output_size(); j++)
				printf("%s ", node.output(j).c_str());
			printf("\n");
		}
		if(!strcmp(node.op_type().c_str(), "Add") && node.input_size() == 2 &&
				i > 0 && node.input(0) == graph.node(i-1).output(0) && n > 0 &&
				((!strcmp(graph.node(i-1).op_type().c_str(), "Conv") && !net->modules[n-1].SpatialConvolution.bias->storage) ||
				(!strcmp(graph.node(i-1).op_type().c_str(), "Gemm") && !net->modules[n-1].Linear.bias->storage)))
		{
			// Special case: we are adding bias to convolution or linear
			if(!strcmp(graph.node(i-1).op_type().c_str(), "Conv"))
			{
				THFloatTensor_free(net->modules[n-1].SpatialConvolution.bias);
				net->modules[n-1].SpatialConvolution.bias = onnx_gettensor(&graph, i, 1);
			} else {
				THFloatTensor_free(net->modules[n-1].Linear.bias);
				net->modules[n-1].Linear.bias = onnx_gettensor(&graph, i, 1);
			}
			free(net->modules[n-1].outputname);
			net->modules[n-1].outputname = strdup(node.output(0).c_str());
			continue;
		}
		if(!strcmp(node.op_type().c_str(), "Split"))
		{
			int from = 0;
			for(j = 0; j < node.output_size(); j++)
			{
				net->modules[n].output = THFloatTensor_new();
				net->modules[n].net = net;
				net->modules[n].updateOutput = nn_Slice_updateOutput;
				net->modules[n].type = MT_Slice;
				net->modules[n].outputname = strdup(node.output(j).c_str());
				net->modules[n].inputs[0] = getoutput(net, node.input(0).c_str());
				net->modules[n].ninputs = 1;
				struct Slice *p = &net->modules[n].Slice;
				p->from = from;
				p->to = p->from + onnx_getint(&graph, i, "split", j);
				from = p->to;
				net->nelem = ++n;
			}
			continue;
		}
		f = getfunction(node.op_type().c_str());
		if(f == -1)
			THError("Unsupported node type %s\n", node.op_type().c_str());
		for(j = 0; j < node.input_size(); j++)
		{
			int k = getoutput(net, node.input(j).c_str());
			if(k >= 0)
			{
				net->modules[n].inputs[net->modules[n].ninputs++] = k;
				if(net->modules[n].ninputs > MAXMODULEINPUTS)
					THError("Maximum number of node inputs exceeded\n");
			}
		}
		if(i == 0 || net->modules[n].ninputs)
		{
			net->modules[n].output = THFloatTensor_new();
			net->modules[n].net = net;
			name2loadf[f].onnxload(&graph, net->modules + n, i);
			net->modules[n].outputname = strdup(node.output(0).c_str());
			net->nelem = ++n;
		}
	}

	google::protobuf::ShutdownProtobufLibrary();

	return net;
}
