?	:y?	 ?@:y?	 ?@!:y?	 ?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC:y?	 ?@?$???֧@1J{?/LV6@A???7??I???sm!@rEagerKernelExecute 0*	U-??/U@2U
Iterator::Model::ParallelMapV2?p!????!?[M??\:@)?p!????1?[M??\:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?*???,??!f4?vP?@)???SVӕ?1?iê&9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??%?`??!E??+??9@)??G????1_<?跆0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?Os?"??!͙??&?"@)?Os?"??1͙??&?"@:Preprocessing2F
Iterator::Model???????!??BA@)E??@J?z?1?m+?s@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?wE𿕬?!9?y?^xP@)76;R}?w?1^?????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??b?du?!a?(?/?@)??b?du?1a?(?/?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapn?r???!)u?	?;@)???$xCZ?1J????C??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 99.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI??u??X@Q??vE0??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?$???֧@?$???֧@!?$???֧@      ??!       "	J{?/LV6@J{?/LV6@!J{?/LV6@*      ??!       2	???7?????7??!???7??:	???sm!@???sm!@!???sm!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??u??X@y??vE0???
"?
kgradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_1/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilterP????S??!P????S??0"?
igradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter?X'0?(??!~?B>??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?ƙg?ɗ?!?<?????0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_8/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter^'j????!Rz?????0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_5/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?<s??l??!?Җ??\??0"?
Wgradient_tape/pokemon-images-semantic-segmentation/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterҦJA6%??!?t?&S??0"?
kgradient_tape/pokemon-images-semantic-segmentation/conv2d_transpose_6/conv2d_transpose/Conv2DBackpropFilterConv2DBackpropFilter?o???7??!??Z??0"R
4pokemon-images-semantic-segmentation/conv2d_8/Conv2DConv2DDM??h??!???H?K??0"o
Fpokemon-images-semantic-segmentation/conv2d_transpose/conv2d_transposeConv2DBackpropInput?0????!???b?,??"q
Hpokemon-images-semantic-segmentation/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput?R]֮??!??ȼ??Q      Y@YxN[@a?K??W@qW?LX@yi<	??Ԃ?"?

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
Refer to the TF2 Profiler FAQb?96.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 