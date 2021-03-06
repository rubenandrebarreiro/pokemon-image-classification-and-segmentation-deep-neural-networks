?	=C8f???@=C8f???@!=C8f???@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC=C8f???@?W?????@1t]?@?)@Ad?&???I??l;m?@rEagerKernelExecute 0*	أp=
X@2U
Iterator::Model::ParallelMapV2ut\????!???q[;@)ut\????1???q[;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?aL?{)??!?M?y?<@)?ZH???1???9&9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatej?drjg??!?u@$??8@)I?s
????1??`??1@:Preprocessing2F
Iterator::Model?2?FY???!ED??qB@)??#bJ$??1T??j!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????z?!?k?G<@)????z?1?k?G<@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\??.?u??!??hl??O@)d!:?z?1??iE?o@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??d?VA??!ӭ?7??<@)??c?n?1??1?HN@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???Ik?!?@? ?@)???Ik?1?@? ?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?5<???X@Q?6???-??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?W?????@?W?????@!?W?????@      ??!       "	t]?@?)@t]?@?)@!t]?@?)@*      ??!       2	d?&???d?&???!d?&???:	??l;m?@??l;m?@!??l;m?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?5<???X@y?6???-???
"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ϥA????!?ϥA????0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Ṕ??!Y?1??0"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????<??!?L? ??0"T
+Adamax/Adamax/update_16/ResourceApplyAdaMaxResourceApplyAdaMax?a%S???!ؕ?h???"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?ze?`??!?>??J???0"]
;pokemon-images-multi-label-classification/activation_7/Relu_FusedConv2D?uE????!??bq"??"]
;pokemon-images-multi-label-classification/activation_6/Relu_FusedConv2D?:??A??!?4;2C???"?
\gradient_tape/pokemon-images-multi-label-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilters~4?X???!??aO???0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput=????!d?`?P
??0"?
[gradient_tape/pokemon-images-multi-label-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput?ؽ\???!??le?	??0Q      Y@Yv??#`@@a????P@q?g(?W@y???2hÕ?"?

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
Refer to the TF2 Profiler FAQb?94.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 