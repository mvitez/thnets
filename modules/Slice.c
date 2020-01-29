#include "../thnets.h"

#ifdef ONNX
void onnxload_Slice(const void *graph, struct module *m, int nodeidx)
{
	m->updateOutput = nn_Slice_updateOutput;
	m->type = MT_Slice;
	struct Slice *p = &m->Slice;
	if(th_debug > 2)
	{
		onnx_printintslist(graph, nodeidx, "axes");
		onnx_printintslist(graph, nodeidx, "starts");
		onnx_printintslist(graph, nodeidx, "ends");
	}
	p->axis = onnx_getint(graph, nodeidx, "axes", 0);
	p->from = onnx_getint(graph, nodeidx, "starts", 0);
	p->to = onnx_getint(graph, nodeidx, "ends", 0);
}
#endif

THNTensor *nn_Slice_updateOutput(struct module *module, THNTensor *input)
{
	struct Slice *p = &module->Slice;
	THNTensor_slice(module->output, input, p->axis, p->from, p->to);
	return module->output;
}
