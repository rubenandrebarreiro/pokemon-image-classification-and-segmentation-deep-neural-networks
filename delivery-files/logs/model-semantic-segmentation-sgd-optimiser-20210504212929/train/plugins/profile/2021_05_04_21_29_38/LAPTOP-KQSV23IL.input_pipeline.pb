	K??3h?@K??3h?@!K??3h?@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCK??3h?@?1ܤ@1)x
?>6@A??ۻ}??I)H4?#@rEagerKernelExecute 0*	???S?-R@2U
Iterator::Model::ParallelMapV2M???D??!v??<@)M???D??1v??<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?>??V??!???Q >@)g??/??1\ƈ? ?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?{?_????!????j6@)k?MG ??1 ???.@:Preprocessing2F
Iterator::ModelJ?%r???!??ly?B@)?W?\T{?15?f?Y"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???H.??!p???!O@)?`"?u?1??mC+;@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??.?t?!??i???@)??.?t?1??i???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorA??h:;i?!rmЬ@?@)A??h:;i?1rmЬ@?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapj?WV????!m?]:??8@)??7h?>^?1f??O@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 98.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI???gP?X@Qƃ?!?W??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?1ܤ@?1ܤ@!?1ܤ@      ??!       "	)x
?>6@)x
?>6@!)x
?>6@*      ??!       2	??ۻ}????ۻ}??!??ۻ}??:	)H4?#@)H4?#@!)H4?#@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q???gP?X@yƃ?!?W??