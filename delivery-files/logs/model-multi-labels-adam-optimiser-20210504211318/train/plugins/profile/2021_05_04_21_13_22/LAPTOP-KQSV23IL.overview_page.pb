?	u?)??@u?)??@!u?)??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCu?)??@X??"'՘@1??+f?(@AfO?sp??I%̴?+[@rEagerKernelExecute 0*	?v???`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatЗ??\4??!?=$?J=@)??P?v0??1&`?^:@:Preprocessing2U
Iterator::Model::ParallelMapV2??,??Ρ?!?L)???9@)??,??Ρ?1?L)???9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??<e5??!??Z?<?>@)?~m?????1*?]?Y/@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{??{?ʔ?!ru)?$.@){??{?ʔ?1ru)?$.@:Preprocessing2F
Iterator::ModelD?.l?V??!?? ?f1@@)ѕT? ??1
b?BH@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip^??a?Q??!9???L?P@)???!??1?g5*?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorF?n?1p?!?d!a_@)F?n?1p?1?d!a_@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??҈?}??!??YU?M@@)ro~?D?d?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI`?"?X@Q}?ϱnw??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	X??"'՘@X??"'՘@!X??"'՘@      ??!       "	??+f?(@??+f?(@!??+f?(@*      ??!       2	fO?sp??fO?sp??!fO?sp??:	%̴?+[@%̴?+[@!%̴?+[@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q`?"?X@y}?ϱnw???
"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter,;?%9:??!,;?%9:??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???+?x??!?~??f???0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?4?d??!??U?o??0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??!x?A??!=c^c????0"L
%Adam/Adam/update_16/ResourceApplyAdamResourceApplyAdam???-v??!splR?<??"]
;pokemon-images-multi-label-classification/activation_6/Relu_FusedConv2D?O,O?+??!n?Q|.???"]
;pokemon-images-multi-label-classification/activation_7/Relu_FusedConv2D~?}K=???!^??%?~??"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??(ɣ?! ?(???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput?	?	ױ??!5JP	V??0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInputt?`X???!$?i!"??0Q      Y@Y͡bAs@@a~?N_?P@q?=a`?V@y#)ڔ???"?

both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?88.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 