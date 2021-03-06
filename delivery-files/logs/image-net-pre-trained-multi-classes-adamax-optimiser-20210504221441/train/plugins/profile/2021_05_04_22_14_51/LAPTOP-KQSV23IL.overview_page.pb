?	R~R???@R~R???@!R~R???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCR~R???@?8b-???@1??&2s%L@A>?$@Mm@I?u?T@rEagerKernelExecute 0*	?K7QۊRA2?
ZIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?Д????@!???^??X@)?Д????@1???^??X@:Preprocessing2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch??ut\??!?*??OTC?)??ut\??1?*??OTC?:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism????켥?!}!???L?)u?)?:??1???ל?2?:Preprocessing2Y
"Iterator::Model::PrivateThreadPoolm?Yg|??!???K?N?)?E'K??k?1???ni?:Preprocessing2?
LIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch::FlatMapUMu???@!???z??X@)b?o?j?1?|??y??:Preprocessing2F
Iterator::Model0??\??!?@vZ?P?)?2?FY?i?1?H?H??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIK?ʅw?X@Q3-N????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?8b-???@?8b-???@!?8b-???@      ??!       "	??&2s%L@??&2s%L@!??&2s%L@*      ??!       2	>?$@Mm@>?$@Mm@!>?$@Mm@:	?u?T@?u?T@!?u?T@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qK?ʅw?X@y3-N?????"?
}gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_4/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??	??!??	??0"T
+Adamax/Adamax/update_65/ResourceApplyAdaMaxResourceApplyAdaMax????NC??!m?wk????"-
IteratorGetNext/_2_Recv???l???!?.;[sô?"?
kgradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_pw_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?SR?????!??τ]???0"?
kgradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_pw_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??}?D$??!!1o?nz??0"?
}gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_9/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterF??5??!e??.? ??0"?
}gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_8/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterq?P\?4??!??M????0"?
~gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_11/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterjl??p!??!?[]?????0"?
~gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_10/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?O&?G???!???
L???0"?
}gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_7/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterk?2č?!Ƹ?????0Q      Y@Y}N??E@a???kT?L@q<???c??y?Y??=?m?"?	
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