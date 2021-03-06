?	??? ?@??? ?@!??? ?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC??? ?@I?F????@1??	?y?(@A????'???Ien??@rEagerKernelExecute 0*	?t?VY@2U
Iterator::Model::ParallelMapV2? Pō[??!J?Y?cS;@)? Pō[??1J?Y?cS;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ݰmQ??!?????5@)6??D.8??1?J?$?2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0?k?????!?[??SP@)?? ?rh??1??r?H?0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateū?m???!??"?6@)a7l[?ِ?1
??v?<0@:Preprocessing2F
Iterator::Modelqs* ??!v?I?jXA@)b?[>??~?1?r?H?u@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?6?De?z?!O?q??@)?6?De?z?1O?q??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap()? ???!?kv[;@)K?P?r?1?rm
?`@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor>?4a??h?!?6??@)>?4a??h?1?6??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noIz?)-?X@Q.??k???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	I?F????@I?F????@!I?F????@      ??!       "	??	?y?(@??	?y?(@!??	?y?(@*      ??!       2	????'???????'???!????'???:	en??@en??@!en??@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qz?)-?X@y.??k????
"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterP  {??!P  {??0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??_?	??!_󲿏B??0"T
+Adamax/Adamax/update_16/ResourceApplyAdaMaxResourceApplyAdaMax|??r???!??^?
??"]
;pokemon-images-multi-class-classification/activation_6/Relu_FusedConv2Du?Mƺ???!???yF??"]
;pokemon-images-multi-class-classification/activation_7/Relu_FusedConv2D????Q???!???A??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?²??y??!:l?"???0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter#ͧ?s??!???????0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput??bۥ??!?c)?\T??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput6%w4????!?H??4h??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput.?E?X.??!ͣ<@[??0Q      Y@Yr?q?+@ar?q?U@q??b???V@y??q$??"?

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
Refer to the TF2 Profiler FAQb?91.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 