?	2=a??԰@2=a??԰@!2=a??԰@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC2=a??԰@{???I??@1cAaP?iK@A?/??ѳ@I???J @rEagerKernelExecute 0*	???F?KNA2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??CR??@!?Օ???X@)??CR??@1?Օ???X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch#??fF???!"?b?&E?)#??fF???1"?b?&E?:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism*S?A?Ѣ?!rv]??TN?)bX9?Ȇ?1??j?X\2?:Preprocessing2F
Iterator::ModelG=D?;???!`ߦBZQ?))??qh?1?Q%????:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMapY?????@!Z?????X@)?f???e?1c? {.e?:Preprocessing2Y
"Iterator::Model::PrivateThreadPool?q6??!D?T?P?)??UJ??b?1N??`??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?契?X@Q9???[??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{???I??@{???I??@!{???I??@      ??!       "	cAaP?iK@cAaP?iK@!cAaP?iK@*      ??!       2	?/??ѳ@?/??ѳ@!?/??ѳ@:	???J @???J @!???J @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?契?X@y9???[???"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_4/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter????Π?!????Π?0"-
IteratorGetNext/_2_Recv??ao?@??!̰?????"?
kgradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_pw_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?;龓?!.???;g??0"?
kgradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_pw_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?z????!h?u?V??0"\
0Adagrad/Adagrad/update_65/ResourceApplyAdagradV2ResourceApplyAdagradV2i?C?+???!n??<??"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_9/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?)??G???!??oy??0"?
~gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_11/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterޛUn?!??!Z3U.?	??0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_7/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?U]???!??????0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_8/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterP?]?ڴ??!?? D ???0"?
~gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_10/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterEѦ?^??!??m?????0Q      Y@Y????c+D@a[4??M@q??E"????y5?[/r?"?	
both?Your program is POTENTIALLY input-bound because 98.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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