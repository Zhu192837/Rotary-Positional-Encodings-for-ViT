—Rotary Positional Encodings (RoPE) is a mechanism that efficiently utilizes position information. It
encodes the absolute position through a rotation matrix and explicitly incorporates the dependency of relative position in
the self-attention mechanism. In recent years, RoPE has shown significant potential in the research of Vision
Transformers (ViTs) as a method to efficiently capture position information. This project is based on the multiple RoPE
variants considered in the recent study: Rotary position embedding for vision transformer. CoRR, abs/2403.13298., and systematically applies and evaluates them on CIFAR-10
datasets. By integrating different RoPE variants into regular Transformer and Performer architectures, we conduct a
comprehensive comparative analysis in terms of classification accuracy, training efficiency, and inference speed. The
results of thsi project show that the Mixed variant of RoPE performs best among the ViT and Performer models. Although
ViT performs better on short sequence tasks such as CIFAR-10, Performer has significant advantages in inference
throughput and computational scalability, making it a more attractive choice in resource-constrained or long sequence
tasks. This study provides valuable guidance on how to choose the appropriate RoPE variant and model architecture
based on task requirements.
