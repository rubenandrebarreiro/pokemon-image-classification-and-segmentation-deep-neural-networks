	??\??0@??\??0@!??\??0@	?J?R+????J?R+???!?J?R+???"?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsL??\??0@??V???@1Ԟ?sbW%@A?c*?ߗ?I?7?k?g@Y[??vN???rEagerKernelExecute 0*	??x?&y[@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat? 3??O??!p????B@)??]Pߢ?1?¶W?@@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?&??0???!???<?`4@)?&??0???1???<?`4@:Preprocessing2U
Iterator::Model::ParallelMapV2???]M???!?/<??0@)???]M???1?/<??0@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??.5B???!n?QۙA@)?t?? ??1????T?+@:Preprocessing2F
Iterator::Model?b.???!?8?/?7@)?k$	???1?e$?z?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{i? ?w??!?q?S@)?:??Kt?1jQ%	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??5"g?!]W??v@)??5"g?1]W??v@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap~?????!v8???A@)?0{?v?Z?1???|????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 17.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?16.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?J?R+???I~??!?lA@Q.????O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??V???@??V???@!??V???@      ??!       "	Ԟ?sbW%@Ԟ?sbW%@!Ԟ?sbW%@*      ??!       2	?c*?ߗ??c*?ߗ?!?c*?ߗ?:	?7?k?g@?7?k?g@!?7?k?g@B      ??!       J	[??vN???[??vN???![??vN???R      ??!       Z	[??vN???[??vN???![??vN???b      ??!       JGPUY?J?R+???b q~??!?lA@y.????O@