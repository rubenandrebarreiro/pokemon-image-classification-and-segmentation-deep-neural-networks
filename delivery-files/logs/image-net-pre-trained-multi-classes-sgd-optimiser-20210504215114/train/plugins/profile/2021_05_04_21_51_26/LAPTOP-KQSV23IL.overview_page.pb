?	?K?AS??@?K?AS??@!?K?AS??@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?K?AS??@?M~????@1??D?k?H@A&:?,B???ISwe?@rEagerKernelExecute 0*	??"??G@2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::Prefetch	?%qVD??!???rZ?N@)	?%qVD??1???rZ?N@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelismra?ri??!r??j??U@)??8G??1b??h8@:Preprocessing2F
Iterator::ModelO<g???!      Y@)ݚt["l?1I??N?@:Preprocessing2Y
"Iterator::Model::PrivateThreadPool????????!Ŗk%W@)????5"h?1???N?{@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIy?}?7?X@Q??? 2??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?M~????@?M~????@!?M~????@      ??!       "	??D?k?H@??D?k?H@!??D?k?H@*      ??!       2	&:?,B???&:?,B???!&:?,B???:	Swe?@Swe?@!Swe?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qy?}?7?X@y??? 2???"?
}gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_4/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?t)hؠ?!?t)hؠ?0"-
IteratorGetNext/_2_Recvoչa??!?? ????"?
kgradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_pw_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter,?ޓ?)??!D ??1=??0"?
kgradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_pw_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterlkfe(??!?Q?4??0"\
,SGD/SGD/update_65/ResourceApplyKerasMomentumResourceApplyKerasMomentum"??_???!hg?»?"?
ggradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_pw_3_bn/FusedBatchNormGradV3FusedBatchNormGradV3ȓ?-ۋ??!??????"?
~gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_10/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??~?6|??!@?ZSǡ??0"?
}gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_9/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter??~?6|??!??R??i??0"?
}gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_8/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterW-K??\??!d?G?Y/??0"?
~gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_11/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter????Y??!?g?M????0Q      Y@Y?wɃg@aC????X@qk??J?X@y $4'?q?"?

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
Refer to the TF2 Profiler FAQb?98.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 