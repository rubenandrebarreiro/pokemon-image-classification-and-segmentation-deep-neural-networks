	?zܷZ?2@?zܷZ?2@!?zܷZ?2@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC?zܷZ?2@?????@1\W?'%@A{-??1??Iur?⎗@rEagerKernelExecute 0*	?x?&1?R@2U
Iterator::Model::ParallelMapV2_%????!??̆??:@)_%????1??̆??:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??I???!?:???@<@)?q?_!??1?s1l?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateS!?????!?ݖ?5?9@)?^??W??1&?Sz͡3@:Preprocessing2F
Iterator::Model??#nk??!0????A@)I?H?]{?1N??|??!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip3k) ???!?綮?P@)??0?*x?1??Z???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice*??F??r?!?n͠Y@)*??F??r?1?n͠Y@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?_=?[?c?!q9f???	@)?_=?[?c?1q9f???	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap	5C?(^??!\?F\?;@)\;Qi[?1?u-?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?23.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?Z??DF@Q]?s{??K@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????@?????@!?????@      ??!       "	\W?'%@\W?'%@!\W?'%@*      ??!       2	{-??1??{-??1??!{-??1??:	ur?⎗@ur?⎗@!ur?⎗@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Z??DF@y]?s{??K@