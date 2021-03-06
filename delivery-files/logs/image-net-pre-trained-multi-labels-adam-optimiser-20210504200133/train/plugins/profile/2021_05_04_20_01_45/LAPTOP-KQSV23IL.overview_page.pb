?	 ????@ ????@! ????@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC ????@???Jkڳ@1?r?w??K@A????5@Iǽ??@rEagerKernelExecute 0*	???ȋNA2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???4[G?@!<?????X@)???4[G?@1<?????X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch:#J{?/??!O?Ѓ?F?):#J{?/??1O?Ѓ?F?:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelismFzQ?_??!㸒?NQ?)??TO???1?)G7?:Preprocessing2F
Iterator::Model.?Ue???!"?|D?T?)!?J?n?1z?o>[??:Preprocessing2Y
"Iterator::Model::PrivateThreadPool?D?
)??!????ЂR?)R臭??l?1~z3??:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap+/??\G?@!??I???X@)?T?	g?f?1ʔr??'?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIR??U?X@Q~kȾ?*??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???Jkڳ@???Jkڳ@!???Jkڳ@      ??!       "	?r?w??K@?r?w??K@!?r?w??K@*      ??!       2	????5@????5@!????5@:	ǽ??@ǽ??@!ǽ??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qR??U?X@y~kȾ?*???"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_4/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter[	/????![	/????0"-
IteratorGetNext/_2_RecvUA?m֕?!????????"L
%Adam/Adam/update_65/ResourceApplyAdamResourceApplyAdamQ?sٛ ??!??kJ??"?
kgradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_pw_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter!S?þ???!?i??5??0"?
kgradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_pw_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter.J?n????!X<o8???0"?
~gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_11/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?2?*??!???o???0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_8/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?2?*??!,???u??0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_7/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?M??'??!*xoCe??0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_9/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??[l????!??=?>T??0"?
~gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_10/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??|????!?ޝ2??0Q      Y@Y|?9KtG@a?ƴ??J@q??~?????y?cɼ?]z?"?	
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