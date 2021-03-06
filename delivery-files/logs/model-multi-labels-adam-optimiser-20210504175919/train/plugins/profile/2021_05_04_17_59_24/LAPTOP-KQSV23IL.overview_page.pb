?	??W;?f?@??W;?f?@!??W;?f?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??W;?f?@]?F?!?@1??8?(@A??Ü???I?a0??@rEagerKernelExecute 0*	??~j?X@2U
Iterator::Model::ParallelMapV2MN?S[??!?5Aj??:@)MN?S[??1?5Aj??:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???4???!3z??)
8@)ʋL?????1?>??4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??E&?ט?!????/9@)????^???1BF????2@:Preprocessing2F
Iterator::Model?????̤?!???hrE@)??X?|??1?S??~?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice? ??x?!?I????@)? ??x?1?I????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???N????!9j???L@)????=v?1?!
?<U@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?4?($i?!
?Ur-}	@)?4?($i?1
?Ur-}	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?#bJ$ћ?!??<?3<@)??? !?g?1?)8?\@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI*/?ۀ?X@Qk?????Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]?F?!?@]?F?!?@!]?F?!?@      ??!       "	??8?(@??8?(@!??8?(@*      ??!       2	??Ü?????Ü???!??Ü???:	?a0??@?a0??@!?a0??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q*/?ۀ?X@yk??????
"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???O?V??!???O?V??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??!N;???!POx??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter=	??~??!??Y??@??0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?4??@e??!"??9??0"L
%Adam/Adam/update_16/ResourceApplyAdamResourceApplyAdam?8?M????!,??Z]??"]
;pokemon-images-multi-label-classification/activation_6/Relu_FusedConv2D?P?L????!C???l???"]
;pokemon-images-multi-label-classification/activation_7/Relu_FusedConv2D???4???!??O?e??"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?J.٣?!?P????0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInputw??*8???!??b/??0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput$?)?ć??!`?Ǣ' ??0Q      Y@Y?m۶m?&@aI?$I?$V@q\?\?W@y`!??`G??"?

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
Refer to the TF2 Profiler FAQb?95.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 