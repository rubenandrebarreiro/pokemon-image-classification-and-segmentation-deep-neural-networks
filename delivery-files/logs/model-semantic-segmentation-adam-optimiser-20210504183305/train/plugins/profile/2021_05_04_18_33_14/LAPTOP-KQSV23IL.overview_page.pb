?	.rO???@.rO???@!.rO???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC.rO???@U?-?a??@1a???`6@AEH?ξ??I?5Φ#?@rEagerKernelExecute 0*R㥛ĀY@)      =2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?K?^I??!??C	{?A@)?t۠?1??|??"@@:Preprocessing2U
Iterator::Model::ParallelMapV2}?!8.???!:>?? ?5@)}?!8.???1:>?? ?5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateJ'L5???!.??/?5@)?u?X???1+???T0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipw;S輲?!1HS?Q@)࠽?x???1/??;j?*@:Preprocessing2F
Iterator::Model?{F"4???!9ߞ???<@)?1??|z?1 ???Z@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??J̳?v?!Ċ ??@)??J̳?v?1Ċ ??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorI?2??f?!2?o?B?@)I?2??f?12?o?B?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapE?Ɵ?l??!??a7@)?/K;5?[?1XH?Şi??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?4?vN?X@Q?????X??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	U?-?a??@U?-?a??@!U?-?a??@      ??!       "	a???`6@a???`6@!a???`6@*      ??!       2	EH?ξ??EH?ξ??!EH?ξ??:	?5Φ#?@?5Φ#?@!?5Φ#?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?4?vN?X@y?????X???
"?
kgradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter?Xuiޫ??!?Xuiޫ??0"?
igradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter??%?5??!s????p??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter$P?t???!?k??? ??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterH?Z>?Ǖ?!??x̒??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?m??]??!?(??j??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?=i???!8???X??0"?
kgradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_6/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterJE0????!?Ĕ?:i??0"R
4pokemon-images-semantic-segmentation/conv2d_8/Conv2DConv2D??"?I??!????vr??0"o
Fpokemon-images-semantic-segmentation/conv2d_transpose/conv2d_transposeConv2DBackpropInput??U8?@??!!sV??"q
Hpokemon-images-semantic-segmentation/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput?G?N????!?]bA]5??Q      Y@YxN[@a?K??W@q(r??&?W@yߞ?????"?

both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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