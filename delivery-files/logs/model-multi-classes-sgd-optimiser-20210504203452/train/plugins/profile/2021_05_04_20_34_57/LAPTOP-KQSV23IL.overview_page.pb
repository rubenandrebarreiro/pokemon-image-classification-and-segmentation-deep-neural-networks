?	U?]y0@U?]y0@!U?]y0@      ??!       "?
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsCU?]y0@?G??[h@1?jQL?$@AI??Q,???I??????@rEagerKernelExecute 0*	????7d@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????e??!???)ZB@)????????1?5?	H@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat]m???{??!ɛ?݂??@)?vöE??1?Sx?3?>@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???I???!!??[AU@)d??u??1?r09r?.@:Preprocessing2U
Iterator::Model::ParallelMapV2?Y,E??!??=?&@)?Y,E??1??=?&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???ip{?!???$ ?@)???ip{?1???$ ?@:Preprocessing2F
Iterator::Model??9Ϙ?!??y ?-@)??o?4(z?1???y??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorr???_c?!????d??)r???_c?1????d??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapbK??z2??!?K???B@)W??x??Y?1??%???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 21.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?15.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??
zw`B@Q-????O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?G??[h@?G??[h@!?G??[h@      ??!       "	?jQL?$@?jQL?$@!?jQL?$@*      ??!       2	I??Q,???I??Q,???!I??Q,???:	??????@??????@!??????@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??
zw`B@y-????O@?
"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltery]?i??!y]?i??0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??؂??!?'??=??0"]
;pokemon-images-multi-class-classification/activation_7/Relu_FusedConv2D??Y?Lj??!̛??????"]
;pokemon-images-multi-class-classification/activation_6/Relu_FusedConv2D|???T??!Һ?????"\
,SGD/SGD/update_16/ResourceApplyKerasMomentumResourceApplyKerasMomentum????:T??!?Q6(??"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?@9JS??!?y\?o???0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_7/Conv2D/Conv2DBackpropInputConv2DBackpropInput?E?????!??0_rc??0"?
[gradient_tape/pokemon-images-multi-class-classification/conv2d_6/Conv2D/Conv2DBackpropInputConv2DBackpropInput???_?2??!zS(?????0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterh?J8?W??!?1?????0"?
\gradient_tape/pokemon-images-multi-class-classification/conv2d_4/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??gm?@??!?*?????0Q      Y@Y?5??P*@a_Cy??U@q??u??T@yܤ??f&??"?
both?Your program is POTENTIALLY input-bound because 21.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?15.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?82.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 