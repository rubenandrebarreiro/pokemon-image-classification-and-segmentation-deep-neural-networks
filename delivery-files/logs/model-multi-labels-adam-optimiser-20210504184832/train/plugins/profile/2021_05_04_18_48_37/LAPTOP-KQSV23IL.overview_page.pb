?	??Q!?@??Q!?@!??Q!?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??Q!?@???Mʘ@1??B??'@A4J??%???Ieo)??@rEagerKernelExecute 0*	??? ?J]@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatu???a???!?'Y??7@)nP???(??1pjw?4@:Preprocessing2U
Iterator::Model::ParallelMapV2???H???!ɫ ??4@)???H???1ɫ ??4@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??@?ε?! 
??D-R@)ZH?????1??v>? 3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?p???i??!??.???8@)C?l搔?1E?3/$1@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????R???!???1~@)????R???1???1~@:Preprocessing2F
Iterator::Model?
b?k_??!?׻E?J;@)^=?1X??1ٰl???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???9?š?!it?g?=@)k?C4??x?1(?n@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor7 !?l?!??x???@)7 !?l?1??x???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?l??;?X@Q6?I%'???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Mʘ@???Mʘ@!???Mʘ@      ??!       "	??B??'@??B??'@!??B??'@*      ??!       2	4J??%???4J??%???!4J??%???:	eo)??@eo)??@!eo)??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?l??;?X@y6?I%'????
"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?6e????!?6e????0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?~ı???!K?qM????0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter[???h??!x?g????0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??݃?2??!??^????0"L
%Adam/Adam/update_16/ResourceApplyAdamResourceApplyAdam????
???!?]?0?E??"]
;pokemon-images-multi-label-classification/activation_6/Relu_FusedConv2D?HaB?ʤ?!?99 ???"]
;pokemon-images-multi-label-classification/activation_7/Relu_FusedConv2D:??lȦ??!J??F?s??"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??? [??!+y?d]???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput!]aϰ??!Ϥ?~??0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput?^1?p???!??w????0Q      Y@Y?m۶m?&@aI?$I?$V@qqFt?W@y}d?????"?

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
Refer to the TF2 Profiler FAQb?95.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 