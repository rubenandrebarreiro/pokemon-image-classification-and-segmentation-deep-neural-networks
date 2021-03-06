?	%????@%????@!%????@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC%????@?U?³@1????_K@A?%?`F"@I???v?@rEagerKernelExecute 0*	?l??)?>A2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??ZӼ??@!ڷS???X@)??ZӼ??@1ڷS???X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch??e???!?l.U?Z?)??e???1?l.U?Z?:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism?M?#~Ū?!?????e?)?u?|?H??1??u?r_P?:Preprocessing2F
Iterator::Model3?????!?k"y.?i?)%xC8y?1y0?m[4?:Preprocessing2Y
"Iterator::Model::PrivateThreadPoola??w}???!??e??Sg?)??D??q?1?m6?n~+?:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap?o	????@!??A??X@)F?Sweg?1??z?"?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?$۪?X@QR??6I??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?U?³@?U?³@!?U?³@      ??!       "	????_K@????_K@!????_K@*      ??!       2	?%?`F"@?%?`F"@!?%?`F"@:	???v?@???v?@!???v?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?$۪?X@yR??6I???"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_4/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?f?±??!?f?±??0"-
IteratorGetNext/_2_Recvo?R?????!?FZ	???"?
kgradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_pw_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter<?B????!3??=???0"?
kgradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_pw_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterճh????!(?M%?ظ?0"\
,SGD/SGD/update_65/ResourceApplyKerasMomentumResourceApplyKerasMomentum-?0?`Ր?!?Za??"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_9/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterU%??j???!OJ??E???0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_8/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?8?Z???!?O??p??0"?
}gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_7/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter\???????!?5M?8_??0"?
~gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_11/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?^q>)Ď?!?K4F{K??0"?
~gradient_tape/multi-label-classification-mobile-net-image-net-weights/conv_dw_10/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterA\?.M??!??P-??0Q      Y@Y???zWO@a?l9??B@q???T????y?N*?\^t?"?	
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
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Turing)(: B 