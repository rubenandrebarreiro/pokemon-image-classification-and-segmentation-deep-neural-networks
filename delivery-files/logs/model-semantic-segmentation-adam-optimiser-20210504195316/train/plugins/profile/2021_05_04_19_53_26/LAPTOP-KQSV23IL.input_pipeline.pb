	:y?	 ?@:y?	 ?@!:y?	 ?@      ??!       "?
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
	?$???֧@?$???֧@!?$???֧@      ??!       "	J{?/LV6@J{?/LV6@!J{?/LV6@*      ??!       2	???7?????7??!???7??:	???sm!@???sm!@!???sm!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??u??X@y??vE0??