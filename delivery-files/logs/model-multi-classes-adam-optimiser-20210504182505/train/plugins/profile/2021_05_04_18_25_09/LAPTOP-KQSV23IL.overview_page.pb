?	^.?;1?0@^.?;1?0@!^.?;1?0@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsC^.?;1?0@>?h???@1qu ?]e%@A%???Ǫ?IQ?+?/@rEagerKernelExecute 0*	?z?G1^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???0????!|??PB@)sJ@L??13 ??B?@@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip;?O Ÿ?!?U?GkT@)????KU??1n
?"K5@:Preprocessing2U
Iterator::Model::ParallelMapV2??T[??!2???)-@)??T[??12???)-@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate߈?Y?h??!??????4@)?oB!??1???Ӕ+@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?1Xq????!p???@)?1Xq????1p???@:Preprocessing2F
Iterator::ModelƦ?B ???!ۨ??R?3@)?p?;z?1????5@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?St$??p?!??{b?}@)?St$??p?1??{b?}@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9*7QKs??!??7e]26@)?
E??S`?1?\?fg??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?17.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI?ϋ}?B@QX0t??O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	>?h???@>?h???@!>?h???@      ??!       "	qu ?]e%@qu ?]e%@!qu ?]e%@*      ??!       2	%???Ǫ?%???Ǫ?!%???Ǫ?:	Q?+?/@Q?+?/@!Q?+?/@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?ϋ}?B@yX0t??O@?
"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??h?ٰ?!??h?ٰ?0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?s:5ۯ??!?O????0"L
%Adam/Adam/update_16/ResourceApplyAdamResourceApplyAdam"F??5??!??v??K??"]
;pokemon-images-multi-class-classification/activation_6/Relu_FusedConv2D|?6????!LD????"]
;pokemon-images-multi-class-classification/activation_7/Relu_FusedConv2DS?)???!??lJ^??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????!A]?K,??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Mz????!??=d)??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput ?t? ???!]E?u????0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?P:/????!z???????0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?9?.????!C?????0Q      Y@YVg?{?*@a5s???U@q ?I??2W@y:6	d???"?
both?Your program is POTENTIALLY input-bound because 18.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?17.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?92.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 