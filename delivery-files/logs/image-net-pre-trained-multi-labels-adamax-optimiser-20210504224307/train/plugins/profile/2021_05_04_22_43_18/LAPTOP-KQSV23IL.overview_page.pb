?	??9???@??9???@!??9???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??9???@9?Z?9|?@1?8?? 6L@A???V?/??I?={.Ss@rEagerKernelExecute 0*	?n??JK@2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetchw.???v??!????F%L@)w.???v??1????F%L@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelismЛ?T[??!図???U@))	??????1?\[;??>@:Preprocessing2F
Iterator::Model??o???!      Y@)T8?T?m?1;	?+?l@:Preprocessing2Y
"Iterator::Model::PrivateThreadPool???????!m?A?1YW@)M?n?k?1n8?o?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIґ?o?X@Qux?????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	9?Z?9|?@9?Z?9|?@!9?Z?9|?@      ??!       "	?8?? 6L@?8?? 6L@!?8?? 6L@*      ??!       2	???V?/?????V?/??!???V?/??:	?={.Ss@?={.Ss@!?={.Ss@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qґ?o?X@yux??????"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_4/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?*Ju????!?*Ju????0"T
+Adamax/Adamax/update_65/ResourceApplyAdaMaxResourceApplyAdaMax?8???!@G ??"-
IteratorGetNext/_2_Recv%?	?H??!)??a??"?
kgradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_pw_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?
(]???!?R??(??0"?
kgradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_pw_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterQ?\????!?<??qн?0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_8/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??i3???!?~6l???0"?
~gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_11/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?%?2W???!(???a???0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_7/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??j??!??v?????0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_9/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter????ho??!m?1!????0"?
~gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_10/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?bT;??!?N?b|???0Q      Y@Yw{!0u(
@a$?~V?.X@qO??A]?X@yBT???m?"?

both?Your program is POTENTIALLY input-bound because 98.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?99.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 