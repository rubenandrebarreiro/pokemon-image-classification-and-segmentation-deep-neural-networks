	B?????1@B?????1@!B?????1@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCB?????1@?:u???@1?Yh?4?$@Aq??|#???I"P??HF@rEagerKernelExecute 0*	W-???`@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata?unڌ??!8*?h??C@)n?????1?q?% ?B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate¿3???!p?j?:@)?0}?!8??1??O?5@:Preprocessing2U
Iterator::Model::ParallelMapV2?1^???!??p9v3@)?1^???1??p9v3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??f?8??!]%q?*?R@)? ?bG???1?oy?c@:Preprocessing2F
Iterator::ModelQ.?_x%??!?j;?T?8@)uv28J^}?1?'+?5@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice\;Qi{?!eX9??@)\;Qi{?1eX9??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ᱟ?R??!cG?i?Y=@)??Z(?l?1i?N???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensornYk(?g?!??;0?@)nYk(?g?1??;0?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 20.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?20.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?lY???D@Qj??UNyM@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?:u???@?:u???@!?:u???@      ??!       "	?Yh?4?$@?Yh?4?$@!?Yh?4?$@*      ??!       2	q??|#???q??|#???!q??|#???:	"P??HF@"P??HF@!"P??HF@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?lY???D@yj??UNyM@