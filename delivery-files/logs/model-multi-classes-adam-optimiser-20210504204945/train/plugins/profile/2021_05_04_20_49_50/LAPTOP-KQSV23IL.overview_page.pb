?	????6?@????6?@!????6?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC????6?@"???Ǚ@1kQLޠ'@Am ]lZ)??Ix` ?C	@rEagerKernelExecute 0*	33333`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?}t??g??!hh?>}<C@)`???Y??1?ȹ??@@:Preprocessing2U
Iterator::Model::ParallelMapV2?`???|??!??&H^L3@)?`???|??1??&H^L3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?&M??y??!4?t??8@)Y????D??1?h{?_2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?? ?K??!???҆eR@)<?.9???1??S1?@:Preprocessing2F
Iterator::Model????:q??!?88??i:@)?? d˂?18?F?v@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice8M?p]??! i???K@)8M?p]??1 i???K@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??~?nx?!?????@)??~?nx?1?????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??E?n???!m?x?0;@)x???Ĭg?1ʑ
]?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI-ti??X@Q???p˲??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	"???Ǚ@"???Ǚ@!"???Ǚ@      ??!       "	kQLޠ'@kQLޠ'@!kQLޠ'@*      ??!       2	m ]lZ)??m ]lZ)??!m ]lZ)??:	x` ?C	@x` ?C	@!x` ?C	@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q-ti??X@y???p˲???	"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterk??|J??!k??|J??0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??ܿ????!26c????0"L
%Adam/Adam/update_16/ResourceApplyAdamResourceApplyAdam5???v??!????M???"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?P?ŋ\??!?PG?p<??0"]
;pokemon-images-multi-class-classification/activation_7/Relu_FusedConv2Dd?c5??!?"????"]
;pokemon-images-multi-class-classification/activation_6/Relu_FusedConv2D?'W?'??!?? ?e??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??E????!ű?? ???0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInputG,8~??!N??^?+??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput?4??_}??!??:\?[??0"]
;pokemon-images-multi-class-classification/activation_2/Relu_FusedConv2D????d??!`?5?+h??Q      Y@YVg?{?*@a5s???U@q???3?W@y0聙˛??"?

both?Your program is POTENTIALLY input-bound because 99.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 