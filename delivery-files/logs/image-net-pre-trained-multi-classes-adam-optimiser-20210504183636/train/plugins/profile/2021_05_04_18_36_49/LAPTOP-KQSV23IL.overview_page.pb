?	?,?=ϳ@?,?=ϳ@!?,?=ϳ@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?,?=ϳ@?#????@1X?QoH@Ah??s???Ic???n@rEagerKernelExecute 0*	*\???XM@2z
CIterator::Model::PrivateThreadPool::MaxIntraOpParallelism::PrefetchCV?zN??!?;?b?E@)CV?zN??1?;?b?E@:Preprocessing2p
9Iterator::Model::PrivateThreadPool::MaxIntraOpParallelism??ݰmQ??!?Mg-??R@)??d?`T??1?%{#>@:Preprocessing2F
Iterator::Model???bE??!      Y@)o??m?~?1J?\?)@:Preprocessing2Y
"Iterator::Model::PrivateThreadPool????-??!??<?U@)???O?~?1???8?)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIRT?X@QQ?V????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?#????@?#????@!?#????@      ??!       "	X?QoH@X?QoH@!X?QoH@*      ??!       2	h??s???h??s???!h??s???:	c???n@c???n@!c???n@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qRT?X@yQ?V?????"?
}gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_4/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?C<?????!?C<?????0"-
IteratorGetNext/_2_RecvpZ\	???!*ϧ}k>??"L
%Adam/Adam/update_65/ResourceApplyAdamResourceApplyAdambM
k ??!?z?ٽ`??"?
kgradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_pw_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter@??}????!?p
????0"?
kgradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_pw_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterq"]??"??!??!1`???0"?
ggradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_pw_3_bn/FusedBatchNormGradV3FusedBatchNormGradV3QVF? ??!.Osm????"?
}gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_8/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter???	?y??!EwY?j???0"?
}gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_7/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilter?? ?#??!$???z??0"_
Fmulti-class-classification-mobile-net-image-net-weights/conv_pad_2/PadPad?|?'?g??!??qc)1??"?
~gradient_tape/multi-class-classification-mobile-net-image-net-weights/conv_dw_11/depthwise/DepthwiseConv2dNativeBackpropFilter#DepthwiseConv2dNativeBackpropFilterg/????!]?t?%???0Q      Y@Y?}?@a??GpX@quq???X@y?K;???z?"?

both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?99.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 