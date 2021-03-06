?	??M+E:?@??M+E:?@!??M+E:?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??M+E:?@.s?Q??@1??/?x((@A???_vO??IgaO;?e@rEagerKernelExecute 0*	?? ?r?_@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?????>??!???^f@@)Ҭl????1?@U?>>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateM??u???!-??̪<@)!<?8b??1?
^庮6@:Preprocessing2U
Iterator::Model::ParallelMapV2??g?,??!?5"=?4@)??g?,??1?5"=?4@:Preprocessing2F
Iterator::Model??+?z???!?????<;@)*??g\8??1C0???
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??uR_???!' "?oA@)2U0*???1?Lc?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceF^???!R???G?@)F^???1R???G?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip-'?􅐷?!??CN?0R@)?????_z?1&?6?[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?1??|j?!???q@)?1??|j?1???q@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?@>(??X@Q????????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	.s?Q??@.s?Q??@!.s?Q??@      ??!       "	??/?x((@??/?x((@!??/?x((@*      ??!       2	???_vO?????_vO??!???_vO??:	gaO;?e@gaO;?e@!gaO;?e@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?@>(??X@y?????????
"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?a?u??!?a?u??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilters+??Y^??!??2bg???0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter? ? J??!?ƃmg???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput???:??!??.<m???0"L
%Adam/Adam/update_16/ResourceApplyAdamResourceApplyAdam?FZ??Ӧ?!E?B3.-??"]
;pokemon-images-multi-label-classification/activation_6/Relu_FusedConv2DFm/;?Ǥ?!:)???"]
;pokemon-images-multi-label-classification/activation_7/Relu_FusedConv2D<?F?¤?!??jÈ^??"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter"? ?P??!:z|C????0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput?l'Y??!???0????0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput?z#?Ԡ?!8a?R]??0Q      Y@Y?m۶m?&@aI?$I?$V@q?pK%I?W@y(?v_A??"?

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
Refer to the TF2 Profiler FAQb?95.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 