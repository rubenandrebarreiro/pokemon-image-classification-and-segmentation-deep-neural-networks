	m;m?v/@m;m?v/@!m;m?v/@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCm;m?v/@'jin?@1???MbX%@A?t??.???I???2?<@rEagerKernelExecute 0*	Zd;߇R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatYvQ????!%?io?K?@)#??^??1d??x?;@:Preprocessing2U
Iterator::Model::ParallelMapV2????)???!????9@)????)???1????9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateh?N?????!?D???;@)?c???H??1MJ???3@:Preprocessing2F
Iterator::Model??8+?&??!?`?:A@)]???az?1?=
?a!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceb??U?u?!C?'z@)b??U?u?1C?'z@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipr?&"???!?O?$?bP@)\?J?p?1???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?????e?!vR?ǃ@)?????e?1vR?ǃ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaps.?Ueߕ?!	ܦ?$?<@)?@fg?;U?1?v??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?13.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?e+?@@QqM?-?P@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	'jin?@'jin?@!'jin?@      ??!       "	???MbX%@???MbX%@!???MbX%@*      ??!       2	?t??.????t??.???!?t??.???:	???2?<@???2?<@!???2?<@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?e+?@@yqM?-?P@