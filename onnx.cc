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

void onnxload_Upsample(const void *graph, struct module *m, int nodeidx);
void onnxload_LSTM(const void *graph, struct module *m, int nodeidx);
void onnxload_GRU(const void *graph, struct module *m, int nodeidx);

static struct {
	const char *name;
	void (*onnxload)(const void *, module *m, int nodeidx);
} name2loadf[] =
{
	{"Conv", onnxload_SpatialConvolution},
	{"ConvTranspose", onnxload_SpatialConvolutionTransposed},
	{"Gemm", onnxload_Linear},
	{"MatMul", onnxload_Linear},
	{"BatchNormalization", onnxload_SpatialBatchNormalization},
	{"MaxPool", onnxload_SpatialMaxPooling},
	{"Relu", onnxload_Threshold},
	{"Dropout", onnxload_Dropout},
	{"Constant", onnxload_Dropout},
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
	{"Slice", onnxload_Slice},
	{"Upsample", onnxload_Upsample},
	{"LSTM", onnxload_LSTM},
	{"GRU", onnxload_GRU}
};

static int getfunction(const char *name)
{
	for(unsigned j = 0; j < sizeof(name2loadf)/sizeof(*name2loadf); j++)
		if(!strcmp(name, name2loadf[j].name))
			return j;
	return -1;
}

static void printtensor(const onnx::TensorProto *t)
{
	printf("%s_Init(", t->DataType_Name(t->data_type()).c_str());
	for(int i = 0; i < t->dims_size(); i++)
	{
		printf("%ld", t->dims(i));
		if(i < t->dims_size() - 1)
			printf(",");
	}
	printf(")");
}

static int isinitializer(const onnx::GraphProto *graph, const string &name)
{
	for(int i = 0; i < graph->initializer_size(); i++)
		if(name == graph->initializer(i).name())
		{
			if(th_debug > 2)
				printtensor(&graph->initializer(i));
			return 1;
		}
	// Not found in initializers, see if it's calculated from other initializers or if it's a shape of an input
	for (int i = 0; i < graph->node_size(); i++)
		if(graph->node(i).output(0) == name)
		{
			if(graph->node(i).op_type() == "Shape")
				return 1;
			for(int j = 0; j < graph->node(i).input_size(); j++)
				if(!isinitializer(graph, graph->node(i).input(j)))
					return 0;
			return 1;
		}
	return 0;
}

static const onnx::TensorProto *getinitializer(const onnx::GraphProto *graph, const string &name)
{
	for(int i = 0; i < graph->initializer_size(); i++)
		if(name == graph->initializer(i).name())
		{
			if(th_debug > 2)
				printtensor(&graph->initializer(i));
			return &graph->initializer(i);
		}
	return 0;
}

extern "C" void onnx_printintslist(const void *graph, int nodeidx, const char *name)
{
	int n = onnx_getint(graph, nodeidx, name, -2);
	printf(" %s=[%d", name, onnx_getint(graph, nodeidx, name, 0));
	for(int i = 1; i < n; i++)
		printf(",%d", onnx_getint(graph, nodeidx, name, i));
	printf("]\n");
}

static THFloatTensor *gettensor(const void *graph, int nodeidx, const char *attrname, int idx)
{
	const onnx::GraphProto *g = (const onnx::GraphProto *)graph;
	for(int i = 0; i < g->node(nodeidx).attribute_size(); i++)
	{
		const onnx::AttributeProto &attr = g->node(nodeidx).attribute(i);
		if(!strcmp(attr.name().c_str(), attrname))
		{
			const onnx::TensorProto *t;
			if(idx == -1)
				t = &attr.t();
			else if(idx == -2)
				return (THFloatTensor *)(size_t)attr.tensors_size();
			else if(idx < attr.tensors_size())
				t = &attr.tensors(idx);
			else return 0;
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
	}
	return 0;
}

extern "C" THFloatTensor *onnx_gettensor(const void *graph, int nodeidx, int inputidx)
{
	const onnx::GraphProto *g = (const onnx::GraphProto *)graph;
	if(inputidx >= g->node(nodeidx).input_size())
		return THFloatTensor_new();
	const onnx::TensorProto *t = getinitializer(g, g->node(nodeidx).input(inputidx));
	if(t)
	{
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
	// Not found in initializers, see if it's calculated
	for (int i = 0; i < g->node_size(); i++)
		if(g->node(i).output(0) == g->node(nodeidx).input(inputidx))
		{
			if(g->node(i).op_type() == "ConstantFill")
			{
				float fill = onnx_getfloat(graph, i, "value", -1);
				if(fill != 0)
					THError("Only zero initial_h and initial_c supported\n");
				return 0;
			}
			if(g->node(i).op_type() == "Constant")
				return gettensor(graph, i, "value", -1);
			struct module m;
			memset(&m, 0, sizeof(m));
			int f = getfunction(g->node(i).op_type().c_str());
			m.output = THFloatTensor_new();
			name2loadf[f].onnxload(graph, &m, i);
			if(f == -1)
				THError("Unsupported node type %s\n", g->node(i).op_type().c_str());
			if(g->node(i).op_type() == "Concat")
			{
				struct network net;
				net.nelem = g->node(i).input_size();
				struct module modules[net.nelem];
				net.modules = modules;
				for(int j = 0; j < net.nelem; j++)
					modules[j].output = onnx_gettensor(graph, i, j);
				m.ConcatTable.net = &net;
				return m.updateOutput(&m, (THFloatTensor *)&m);
			} else return m.updateOutput(&m, onnx_gettensor(graph, i, 0));
		}
	return 0;
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
			if(idx == -2)
				return attr.ints_size();
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
			if(idx == -2)
				return attr.floats_size();
			if(idx < attr.floats_size())
				return attr.floats(idx);
			return 0;
		}
	}
	return 0;
}

extern "C" const char *onnx_getstring(const void *graph, int nodeidx, const char *attrname, int idx)
{
	const onnx::GraphProto *g = (const onnx::GraphProto *)graph;
	for(int i = 0; i < g->node(nodeidx).attribute_size(); i++)
	{
		const onnx::AttributeProto &attr = g->node(nodeidx).attribute(i);
		if(!strcmp(attr.name().c_str(), attrname))
		{
			if(idx == -1)
				return attr.s().c_str();
			if(idx == -2)
				return (const char *)(size_t)attr.strings_size();
			if(idx < attr.strings_size())
				return attr.strings(idx).c_str();
			return 0;
		}
	}
	return 0;
}

void onnxload_Upsample(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = 0; // Not implemented
	m->type = MT_Upsample;
	struct Upsample *p = &m->Upsample;
	p->width_scale = onnx_getfloat(graph, nodeidx, "width_scale", -1);
	p->height_scale = onnx_getfloat(graph, nodeidx, "height_scale", -1);
}

THFloatTensor *notimplemented(struct module *m, THFloatTensor *t)
{
	printf("Not implemented\n");
	return t;
}

void onnxload_LSTM(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = notimplemented;
	m->type = MT_LSTM;
	struct LSTM *p = &m->LSTM;
	p->W = onnx_gettensor(graph, nodeidx, 1);
	p->R = onnx_gettensor(graph, nodeidx, 2);
	p->B = onnx_gettensor(graph, nodeidx, 3);
}

void onnxload_GRU(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = notimplemented;
	m->type = MT_GRU;
	struct GRU *p = &m->GRU;
	p->W = onnx_gettensor(graph, nodeidx, 1);
	p->R = onnx_gettensor(graph, nodeidx, 2);
	p->B = onnx_gettensor(graph, nodeidx, 3);
}

static int getoutput(struct network *net, const char *name)
{
	for(int i = 0; i < net->nelem; i++)
		if(!strcmp(net->modules[i].outputname, name))
			return i;
	return -1;
}

static void printop(const onnx::GraphProto *graph, const onnx::NodeProto *node)
{
	if(th_debug > 1)
	{
		printf("%s (%s): ", node->op_type().c_str(), node->name().c_str());
		for(int j = 0; j < node->input_size(); j++)
			if(!isinitializer(graph, node->input(j)))
				printf("%s ", node->input(j).c_str());
			else printf("'%s' ", node->input(j).c_str());
		printf("-> ");
		for(int j = 0; j < node->output_size(); j++)
			printf("%s ", node->output(j).c_str());
		printf("\n");
	}
}

static int isconstant(const onnx::GraphProto *graph, const onnx::NodeProto *node)
{
	if(node->op_type() == "Shape")
		return 1;
	for(int j = 0; j < node->input_size(); j++)
		if(!isinitializer(graph, node->input(j)))
			return 0;
		else return 1;
	return 1;
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
		printop(&graph, &node);
		if(!strcmp(node.op_type().c_str(), "Add") && node.input_size() == 2)
		{
			int j;
			for(j = 0; j < n; j++)
				if(!strcmp(node.input(0).c_str(), net->modules[j].outputname))
				{
					// Special case: we are adding bias to convolution or linear
					if(net->modules[j].type == MT_SpatialConvolutionVirtMM && !net->modules[j].SpatialConvolution.bias->storage)
					{
						THFloatTensor_free(net->modules[j].SpatialConvolution.bias);
						net->modules[j].SpatialConvolution.bias = onnx_gettensor(&graph, i, 1);
						free(net->modules[j].outputname);
						net->modules[j].outputname = strdup(node.output(0).c_str());
						break;
					} else if(net->modules[j].type == MT_Linear && !net->modules[j].Linear.bias->storage)
					{
						THFloatTensor_free(net->modules[j].Linear.bias);
						net->modules[j].Linear.bias = onnx_gettensor(&graph, i, 1);
						free(net->modules[j].outputname);
						net->modules[j].outputname = strdup(node.output(0).c_str());
						break;
					}
				}
			if(j < n)
				continue;
		}
		if(i+1 < graph.node_size() && !strcmp(node.op_type().c_str(), "Pad") &&
			!strcmp(graph.node(i+1).op_type().c_str(), "Conv") &&
			node.output(0) == graph.node(i+1).input(0))
		{
			// Special case, padding followed by convolution
			printop(&graph, &graph.node(i+1));
			net->modules[n].output = THFloatTensor_new();
			net->modules[n].net = net;
			onnxload_SpatialConvolution(&graph, net->modules + n, i+1);
			net->modules[n].outputname = strdup(node.output(0).c_str());
			const char *mode = onnx_getstring(&graph, i, "mode", -1);
			if(mode)
			{
				if(!strcmp(mode, "reflect"))
					net->modules[n].SpatialConvolution.refl_pad = 1;
				else if(*mode && strcmp(mode, "constant"))
					THError("Unsupported padding type %s\n", mode);
			}
			if(net->modules[n].SpatialConvolution.padH || net->modules[n].SpatialConvolution.padW)
				THError("Double padding not supported\n");
			net->modules[n].SpatialConvolution.padH = onnx_getint(&graph, i, "pads", 2);
			net->modules[n].SpatialConvolution.padW = onnx_getint(&graph, i, "pads", 3);
			net->modules[n].SpatialConvolution.padH2 = onnx_getint(&graph, i, "pads", 6);
			net->modules[n].SpatialConvolution.padW2 = onnx_getint(&graph, i, "pads", 7);
			free(net->modules[n].outputname);
			net->modules[n].outputname = strdup(graph.node(i+1).output(0).c_str());

			int k = getoutput(net, node.input(0).c_str());
			if(k >= 0)
				net->modules[n].inputs[net->modules[n].ninputs++] = k;

			net->nelem = ++n;
			i++;
			continue;
		}
		if(i+1 < graph.node_size() && !strcmp(node.op_type().c_str(), "Pad") &&
			!strcmp(graph.node(i+1).op_type().c_str(), "AveragePool") &&
			node.output(0) == graph.node(i+1).input(0))
		{
			// Special case, padding followed by average pooling
			printop(&graph, &graph.node(i+1));
			net->modules[n].output = THFloatTensor_new();
			net->modules[n].net = net;
			onnxload_SpatialAveragePooling(&graph, net->modules + n, i+1);
			net->modules[n].outputname = strdup(node.output(0).c_str());
			const char *mode = onnx_getstring(&graph, i, "mode", -1);
			if(mode)
			{
				if(!strcmp(mode, "reflect"))
					THError("Unsupported padding type %s followed by average pooling\n", mode);
				else if(*mode && strcmp(mode, "constant"))
					THError("Unsupported padding type %s\n", mode);
			}
			if(net->modules[n].SpatialAveragePooling.padH || net->modules[n].SpatialAveragePooling.padW)
				THError("Double padding not supported\n");
			net->modules[n].SpatialAveragePooling.padH = onnx_getint(&graph, i, "pads", 2);
			net->modules[n].SpatialAveragePooling.padW = onnx_getint(&graph, i, "pads", 3);
			net->modules[n].SpatialAveragePooling.padH2 = onnx_getint(&graph, i, "pads", 6);
			net->modules[n].SpatialAveragePooling.padW2 = onnx_getint(&graph, i, "pads", 7);
			free(net->modules[n].outputname);
			net->modules[n].outputname = strdup(graph.node(i+1).output(0).c_str());

			int k = getoutput(net, node.input(0).c_str());
			if(k >= 0)
				net->modules[n].inputs[net->modules[n].ninputs++] = k;

			net->nelem = ++n;
			i++;
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
		if(!strcmp(node.op_type().c_str(), "Shape") || !strcmp(node.op_type().c_str(), "Squeeze") ||
			!strcmp(node.op_type().c_str(), "Unsqueeze"))
			continue;
		f = getfunction(node.op_type().c_str());
		if(f == -1)
		{
			fprintf(stderr, "WARNING: Unsupported node type %s, substituting with dropout\n", node.op_type().c_str());
			f = getfunction("Dropout");
			//THError("Unsupported node type %s\n", node.op_type().c_str());
		}
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
		if(!isconstant(&graph, &node))
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
