?	8??E?@8??E?@!8??E?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC8??E?@Ѱue?@1???Ŋr(@A??S??IAJ?i@rEagerKernelExecute 0*	5^?I`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatC?B?Y???!???O?>@)M?d??7??1??SpB:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate&?(??=??!?B<??>@)???P?v??1?[?j?9@:Preprocessing2U
Iterator::Model::ParallelMapV2?j?TQ??!??6	?5@)?j?TQ??1??6	?5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?????U??!??r???Q@)?W歺??1??럾 @:Preprocessing2F
Iterator::Model????????!0?5:??<@)I???????185`??@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??[?d8~?!??6Es@)??[?d8~?1??6Es@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????<,t?!??3??@)????<,t?1??3??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????????!)^ ???@@)g|_\??f?1???O_g@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?:??p?X@Q???????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ѱue?@Ѱue?@!Ѱue?@      ??!       "	???Ŋr(@???Ŋr(@!???Ŋr(@*      ??!       2	??S????S??!??S??:	AJ?i@AJ?i@!AJ?i@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?:??p?X@y????????
"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter^?x06??!^?x06??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterAѝ????!?Z???V??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterM?ZLmb??!?ָ@???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput:p?d??!?:?a??0"L
%Adam/Adam/update_16/ResourceApplyAdamResourceApplyAdam3???????!??a???"]
;pokemon-images-multi-label-classification/activation_7/Relu_FusedConv2Dh6??????!??R????"]
;pokemon-images-multi-label-classification/activation_6/Relu_FusedConv2D?Wt??!?=`d???"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter"7?ͣ?!"'#??0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput?lQ=???!??O?C???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput?	?ۥؠ?!?8ʌXZ??0Q      Y@Y?m۶m?&@aI?$I?$V@q ?ZmlW@y?????[??"?

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
Refer to the TF2 Profiler FAQb?93.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 