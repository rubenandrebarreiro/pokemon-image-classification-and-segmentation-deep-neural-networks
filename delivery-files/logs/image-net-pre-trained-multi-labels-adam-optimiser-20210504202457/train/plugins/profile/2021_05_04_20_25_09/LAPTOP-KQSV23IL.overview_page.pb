?	it??R?@it??R?@!it??R?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCit??R?@?Z??@1K??F>QK@AD?l???@I??O?m?@rEagerKernelExecute 0*	??SK$?RA2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorK?r ?@!.=$??X@)K?r ?@1.=$??X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetchd??A??!?:??;?)d??A??1?:??;?:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelismτ&?%??!??Z??K?)??d9	??1????;?:Preprocessing2Y
"Iterator::Model::PrivateThreadPool?J???!b5?5A}P?)	???W?1A??C??$?:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap?W? ?@!@?V??X@)?9???z?1٫??v?!?:Preprocessing2F
Iterator::Model?e?ت?!?!??T?Q?)?"??l?14Ğ#;??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?Dt?ʼX@QR???L???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?Z??@?Z??@!?Z??@      ??!       "	K??F>QK@K??F>QK@!K??F>QK@*      ??!       2	D?l???@D?l???@!D?l???@:	??O?m?@??O?m?@!??O?m?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Dt?ʼX@yR???L????"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_4/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter=?b?{e??!=?b?{e??0"-
IteratorGetNext/_2_Recv????T???!b#?Nh???"L
%Adam/Adam/update_65/ResourceApplyAdamResourceApplyAdam@??3????!nt[???"?
kgradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_pw_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterV܀??Ó?!?.?Զ?0"?
kgradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_pw_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter;4?[???!%?֒??0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_9/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterO?j??;??!/*?WRz??0"?
~gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_10/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?dl????!a?G+????0"?
~gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_11/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter.?F1???!????????0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_8/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?\???n??!p????}??0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_7/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilteru?_H6??!???"Na??0Q      Y@Y|?9KtG@a?ƴ??J@qWܼ2~,??y?N5X??n?"?	
both?Your program is POTENTIALLY input-bound because 98.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 