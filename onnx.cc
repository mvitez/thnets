#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include "onnx.pb.h"
#include "thnets.h"

using namespace std;
using namespace google::protobuf::io;

void onnxload_Upsample(const void *graph, struct module *m, int nodeidx);
void onnxload_LSTM(const void *graph, struct module *m, int nodeidx);
void onnxload_GRU(const void *graph, struct module *m, int nodeidx);
void onnxload_Unsqueeze(const void *graph, struct module *m, int nodeidx);
void onnxload_Squeeze(const void *graph, struct module *m, int nodeidx);
void onnxload_Transpose(const void *graph, struct module *m, int nodeidx);

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
	{"Mul", onnxload_SpatialBatchNormalization},
	{"AveragePool", onnxload_SpatialAveragePooling},
	{"GlobalAveragePool", onnxload_SpatialAveragePooling},
	{"ReduceMean", onnxload_SpatialAveragePooling},
	{"Concat", onnxload_Concat},
	{"Max", onnxload_Cmax},
	{"Slice", onnxload_Slice},
	{"Upsample", onnxload_Upsample},
	{"LSTM", onnxload_LSTM},
	{"GRU", onnxload_GRU},
	{"Unsqueeze", onnxload_Unsqueeze},
	{"Squeeze", onnxload_Squeeze},
	{"Sigmoid", onnxload_Sigmoid},
	{"Tanh", onnxload_Tanh},
	{"Transpose", onnxload_Transpose}
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

void gettensordata(THFloatTensor *tdst, const onnx::TensorProto *tsrc)
{
	float *ddata = tdst->storage->data;
	if(tsrc->has_raw_data())
		memcpy(ddata, tsrc->raw_data().c_str(), tsrc->raw_data().length());
	else {
		int i, n = tsrc->float_data_size();
		for(i = 0; i < n; i++)
			ddata[i] = tsrc->float_data(i);
	}
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
			gettensordata(t1, t);
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
		gettensordata(t1, t);
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
			if(f == -1)
				THError("Unsupported node type %s\n", g->node(i).op_type().c_str());
			name2loadf[f].onnxload(graph, &m, i);
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

extern "C" THFloatTensor *onnx_getshapetensor(const void *graph, int nodeidx, int inputidx)
{
	const onnx::GraphProto *g = (const onnx::GraphProto *)graph;
	if(inputidx >= g->node(nodeidx).input_size())
		return THFloatTensor_new();
	const onnx::TensorProto *t = getinitializer(g, g->node(nodeidx).input(inputidx));
	// Not found in initializers, see if it's a constant
	if(!t)
	{
		for (int i = 0; i < g->node_size(); i++)
			if(g->node(i).output(0) == g->node(nodeidx).input(inputidx))
			{
				if(g->node(i).op_type() == "Constant")
				{
					for(int j = 0; j < g->node(i).attribute_size(); j++)
					{
						const onnx::AttributeProto &attr = g->node(i).attribute(j);
						if(!strcmp(attr.name().c_str(), "value"))
						{
							t = &attr.t();
							break;
						}
					}
				}
				break;
			}
	}
	if(t)
	{
		if(t->data_type() != 7)
			THError("Only int64 tensors are supported for shapes, got data_type %d for %s\n", t->data_type(), t->name().c_str());
		if(t->dims_size() != 1)
			THError("Shape tensors must have dimension 1, this one has dimension %d\n", t->dims_size());
		THFloatTensor *t1 = THFloatTensor_new();
		int64_t sizes[4], total = 1;
		int64_t *data = (int64_t *)t->raw_data().c_str();
		for(int i = 0; i < t->dims(0); i++)
		{
			sizes[i] = data && data[i] ? data[i] : t->int64_data(i);
			total *= sizes[i];
		}
		THFloatTensor_resize(t1, sizes, t->dims(0));
		return t1;
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

THFloatTensor *notimplemented(struct module *m, THFloatTensor *t)
{
	printf("Not implemented\n");
	return t;
}

void onnxload_Upsample(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = notimplemented;
	m->type = MT_Upsample;
	struct Upsample *p = &m->Upsample;
	p->width_scale = onnx_getfloat(graph, nodeidx, "width_scale", -1);
	p->height_scale = onnx_getfloat(graph, nodeidx, "height_scale", -1);
	if(p->width_scale == 0 && p->height_scale == 0)
	{
		p->height_scale = onnx_getfloat(graph, nodeidx, "scales", 2);
		p->width_scale = onnx_getfloat(graph, nodeidx, "scales", 3);
	}
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

THFloatTensor *updateOutput_Unsqueeze(struct module *m, THFloatTensor *t)
{
	struct Squeeze *p = &m->Squeeze;
	int i, idx;
	THFloatTensor *t2 = THFloatTensor_new();
	THFloatTensor_set(t2, t);
	for(i = 0; i < p->naxes && t->nDimension < 4; i++)
	{
		idx = p->axes[i];
		memmove(t2->size+idx+1, t2->size+idx, (t2->nDimension - idx) * sizeof(t2->size[0]));
		memmove(t2->stride+idx+1, t2->stride+idx, (t2->nDimension - idx) * sizeof(t2->stride[0]));
		t2->size[idx] = 1;
		t2->stride[idx] = t2->stride[idx+1];
		t2->nDimension++;
	}
	return t2;
}

void onnxload_Unsqueeze(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = updateOutput_Unsqueeze;
	m->type = MT_Unsqueeze;
	struct Squeeze *p = &m->Squeeze;
	p->naxes = onnx_getint(graph, nodeidx, "axes", -2);
	for(int i = 0; i < p->naxes && i < 4; i++)
		p->axes[i] = onnx_getint(graph, nodeidx, "axes", i);
}

THFloatTensor *updateOutput_Squeeze(struct module *m, THFloatTensor *t)
{
	int i, idx;
	struct Squeeze *p = &m->Squeeze;
	if(p->naxes == 0)
		return THFloatTensor_squeeze(t);
	THFloatTensor *t2 = THFloatTensor_new();
	THFloatTensor_set(t2, t);
	for(i = p->naxes-1; i >= 0; i--)
	{
		idx = p->axes[i];
		if(t2->size[idx] != 1)
			THError("Squeezing non unitary axis %d (size=%ld)\n", idx, t2->size[idx]);
		memmove(t2->size+idx, t2->size+idx+1, (t2->nDimension - (idx+1)) * sizeof(t2->size[0]));
		memmove(t2->stride+idx, t2->stride+idx+1, (t2->nDimension - (idx+1)) * sizeof(t2->stride[0]));
		t2->nDimension--;
	}
	return t2;
}

void onnxload_Squeeze(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = updateOutput_Squeeze;
	m->type = MT_Squeeze;
	struct Squeeze *p = &m->Squeeze;
	p->naxes = onnx_getint(graph, nodeidx, "axes", -2);
	for(int i = 0; i < p->naxes && i < 4; i++)
		p->axes[i] = onnx_getint(graph, nodeidx, "axes", i);
}

THFloatTensor *updateOutput_Transpose(struct module *m, THFloatTensor *t)
{
	THFloatTensor_transpose(m->output, t, 0, 1);
	return m->output;
}

void onnxload_Transpose(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = updateOutput_Transpose;
	m->type = MT_Transpose;
}

static int getoutput(struct network *net, const char *name)
{
	for(int i = 0; i < net->nelem; i++)
	{
		if(!strcmp(net->modules[i].outputname, name))
			return i;
	}
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

static void absorb_bn(struct network *net, int bnidx, int cidx)
{
	struct module *convm = net->modules + cidx;
	struct module *m = net->modules + cidx + 1;
	struct SpatialBatchNormalization *bn = &m->SpatialBatchNormalization;
	int n = bn->running_mean->size[0];
	float *running_mean = THFloatTensor_data(bn->running_mean);
	float *running_var = THFloatTensor_data(bn->running_var);
	float *w_bn = THFloatTensor_data(bn->weight);
	float *b_bn = THFloatTensor_data(bn->bias);
	float eps = bn->eps;
	THFloatTensor *tbias, *tweight;

	if(convm->type == MT_SpatialFullConvolution)
	{
		tbias = convm->SpatialFullConvolution.bias;
		tweight = convm->SpatialFullConvolution.weight;
	} else {
		tbias = convm->SpatialConvolution.bias;
		tweight = convm->SpatialConvolution.weight;
	}

	if(tbias->nDimension == 0)
	{
		THFloatTensor_resize1d(tbias, n);
		memset(THFloatTensor_data(tbias), 0, n * sizeof(float));
	}
	float *bias = THFloatTensor_data(tbias);
	if(convm->type == MT_SpatialFullConvolution)
	{
		// Here output planes are in index 1, so we need to loops
		for(int ii = 0; ii < tweight->size[0]; ii++)
		{
			THFloatTensor *weight2 = THFloatTensor_newSelect(tweight, 0, ii);
			for(int i = 0; i < n; i++)
			{
				THFloatTensor *weight = THFloatTensor_newSelect(weight2, 0, i);
				float *w = THFloatTensor_data(weight);

				float invstd = 1 / sqrtf(running_var[i] + eps);
				int m = THFloatTensor_nElement(weight);
				for(int j = 0; j < m; j++)
					w[j] *= invstd;
				if(w_bn && b_bn)
					for(int j = 0; j < m; j++)
						w[j] *= w_bn[i];
				THFloatTensor_free(weight);
			}
			THFloatTensor_free(weight2);
		}
		for(int i = 0; i < n; i++)
		{
			float invstd = 1 / sqrtf(running_var[i] + eps);
			bias[i] = (bias[i] - running_mean[i]) * invstd;
			if(w_bn && b_bn)
				bias[i] = bias[i] * w_bn[i] + b_bn[i];
		}
	} else {
		for(int i = 0; i < n; i++)
		{
			THFloatTensor *weight = THFloatTensor_newSelect(tweight, 0, i);
			float *w = THFloatTensor_data(weight);

			if(running_mean && running_var)
			{
				float invstd = 1 / sqrtf(running_var[i] + eps);
				int m = THFloatTensor_nElement(weight);
				for(int j = 0; j < m; j++)
					w[j] *= invstd;
				bias[i] = (bias[i] - running_mean[i]) * invstd;
			}
			int m = THFloatTensor_nElement(weight);
			if(w_bn && b_bn)
			{
				for(int j = 0; j < m; j++)
					w[j] *= w_bn[i];
				bias[i] = bias[i] * w_bn[i] + b_bn[i];
			} else if(w_bn)
			{
				for(int j = 0; j < m; j++)
					w[j] *= w_bn[i];
				bias[i] = bias[i] * w_bn[i];
			} else if(b_bn)
				bias[i] += b_bn[i];
			THFloatTensor_free(weight);
		}
	}
	// Free unused tensors
	THFloatTensor_free(m->SpatialBatchNormalization.running_mean);
	THFloatTensor_free(m->SpatialBatchNormalization.running_var);
	THFloatTensor_free(m->SpatialBatchNormalization.weight);
	THFloatTensor_free(m->SpatialBatchNormalization.bias);
	THFloatTensor_free(m->output);
	free(convm->outputname);
	convm->outputname = m->outputname;
	memmove(net->modules + cidx + 1, net->modules + cidx + 2, (net->nelem - (cidx + 2)) * sizeof(net->modules[0]));
	net->nelem--;
	memset(net->modules + net->nelem, 0, sizeof(net->modules[0]));
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
		if(!strcmp(node.op_type().c_str(), "Add") && node.input_size() == 2 &&
			(isinitializer(&graph, node.input(0)) || isinitializer(&graph, node.input(1))))
		{
			int init;
			if(isinitializer(&graph, node.input(0)) && !isinitializer(&graph, node.input(1)))
				init = 0;
			else if(!isinitializer(&graph, node.input(0)) && isinitializer(&graph, node.input(1)))
				init = 1;
			else init = -1;
			int j;
			if(init >= 0)
			{
				for(j = 0; j < n; j++)
					if(!strcmp(node.input(!init).c_str(), net->modules[j].outputname))
					{
						// Special case: we are adding bias to convolution or linear
						if(net->modules[j].type == MT_SpatialConvolutionVirtMM)
						{
							if(net->modules[j].SpatialConvolution.bias->storage)
							{
								THFloatTensor *t = onnx_gettensor(&graph, i, init);
								float *d = THFloatTensor_data(net->modules[j].SpatialConvolution.bias);
								float *s = THFloatTensor_data(t);
								int n = THFloatTensor_nElement(net->modules[j].SpatialConvolution.bias);
								if (THFloatTensor_nElement(t) != n)
									THError("Number of elements mismatch in Add operation (add %d to %d)\n", THFloatTensor_nElement(t), n);
								for(int i = 0; i < n; i++)
									d[i] += s[i];
							} else {
								THFloatTensor_free(net->modules[j].SpatialConvolution.bias);
								net->modules[j].SpatialConvolution.bias = onnx_gettensor(&graph, i, init);
							}
							free(net->modules[j].outputname);
							net->modules[j].outputname = strdup(node.output(0).c_str());
							break;
						} else if(net->modules[j].type == MT_Linear)
						{
							if(net->modules[j].Linear.bias->storage)
							{
								THFloatTensor *t = onnx_gettensor(&graph, i, init);
								float *d = THFloatTensor_data(net->modules[j].Linear.bias);
								float *s = THFloatTensor_data(t);
								int n = THFloatTensor_nElement(net->modules[j].Linear.bias);
								if (THFloatTensor_nElement(t) != n)
									THError("Number of elements mismatch in Add operation (add %d to %d)\n", THFloatTensor_nElement(t), n);
								for(int i = 0; i < n; i++)
									d[i] += s[i];
							} else {
								THFloatTensor *bias = onnx_gettensor(&graph, i, init);
								THFloatTensor_free(net->modules[j].Linear.bias);
								net->modules[j].Linear.bias = THFloatTensor_squeeze(bias);
								THFloatTensor_free(bias);
							}
							free(net->modules[j].outputname);
							net->modules[j].outputname = strdup(node.output(0).c_str());
							break;
						}
					}
				if(j < n)
					continue;
			}
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
		else if(i+2 < graph.node_size() && !strcmp(node.op_type().c_str(), "Pad") &&
			!strcmp(graph.node(i+1).op_type().c_str(), "Transpose") && //initial transpose in tensorflow models
			!strcmp(graph.node(i+2).op_type().c_str(), "Conv") &&
			node.output(0) == graph.node(i+1).input(0) && //these 3 layes are in a sequence
			graph.node(i+1).output(0) == graph.node(i+2).input(0))
		{
			// Special case, padding followed by transpose and convolution
			printop(&graph, &graph.node(i+1));
			printop(&graph, &graph.node(i+2));
			net->modules[n].output = THFloatTensor_new();
			net->modules[n].net = net;
			onnxload_SpatialConvolution(&graph, net->modules + n, i+2);
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
			net->modules[n].SpatialConvolution.padH = onnx_getint(&graph, i, "pads", 1);//transposed the dimensions
			net->modules[n].SpatialConvolution.padW = onnx_getint(&graph, i, "pads", 2);
			net->modules[n].SpatialConvolution.padH2 = onnx_getint(&graph, i, "pads", 5);
			net->modules[n].SpatialConvolution.padW2 = onnx_getint(&graph, i, "pads", 6);
			free(net->modules[n].outputname);
			net->modules[n].outputname = strdup(graph.node(i+2).output(0).c_str());

			int k = getoutput(net, node.input(0).c_str());
			if(k >= 0)
				net->modules[n].inputs[net->modules[n].ninputs++] = k;

			net->nelem = ++n;
			i+=2;
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
		if(!strcmp(node.op_type().c_str(), "Shape"))
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
		if(net->modules[net->nelem-1].type == MT_SpatialBatchNormalization &&
			net->modules[net->nelem-1].inputs[0] >= 0 &&
			(net->modules[net->modules[net->nelem-1].inputs[0]].type == MT_SpatialConvolutionVirtMM ||
			net->modules[net->modules[net->nelem-1].inputs[0]].type == MT_SpatialFullConvolution))
		{
			absorb_bn(net, net->nelem-1, net->modules[net->nelem-1].inputs[0]);
			n--;
		}
	}

	google::protobuf::ShutdownProtobufLibrary();

	return net;
}
