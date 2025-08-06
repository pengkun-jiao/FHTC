# From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning
Repo for paper ["From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning"](https://arxiv.org/pdf/2411.12787)

## News
  📢 [2025-06-26] This work is accepted by ICCV 2025!


## Framework
Overview of our two key components: Visual Cue Enhancement (VCE) module enhances high-level anchor features by aggregating local information from multi-level feature maps. The Dual Low-Rank Adaptation (Dual-LoRA) module projects the input feature into two low-rank subspaces: one for stable holistic domain knowledge (skill space) learning and the other for instruction condition (task space) learning. Dual-LoRA modules is integrated into the LLM’s query and value projection layers for efficiency.
![20991754463329_ pic](https://github.com/user-attachments/assets/f0aad39e-1b3c-4c30-a0a4-1fbc19e7d5d8)


## Comparsion with Existing LoRA-MoE Methods
Comparison between the proposed Dual-LoRA and existing methods. (a) Mainstream MLLMs, e.g. LLaVA and QwenV, project the high-level visual feature map. (b) Our Visual Cue Enhancement module enhances high-level features by aggregating local information from multi-level feature maps. LoRA-MoE methods mitigate data conflicts by enabling localized responses, i.e., experts activation, tailored to several activation strategies: (c) Sparse Activation, where only the top-k experts are activated; (d) Dense Activation, where all experts are activated; and Rectified Activation, where multiple heterogeneous experts are dynamically activated. In contrast, our (f) Dual Low-Rank Adaptation rectifies a holistic knowledge (skill) space with an additional task space, which is fully differentiable, capable of learning any local response, and more structurally efficient.
![21001754463343_ pic](https://github.com/user-attachments/assets/b39d1280-48b6-4a2f-aee1-691182f24cd7)



## Local Enhanced Visual Cues
Feature Map Visualization of Enhanced Visual Cue with VCE: (a) Enhanced Visual Cue emphasize key areas in food imagery, such as textures, garnishes, and toppings. (b) The cues highlight important details: text for readability, the cheetah-prey interaction, and the woman’s face and Oscar trophy on stage.
![20971754463184_ pic](https://github.com/user-attachments/assets/1a586fc3-56ce-49f0-91ac-b0c38f8530aa)


## Local Enhanced Adaption Space
Panels (a), (b), and (c) represent feature visualizations for the recipe generation task where: (a) the distributions of feature outputs for the holistic skill space, (b) the distributions for the rectified skill space, and (c) the entropy of the holistic skill space (blue line) and the rectified skill space (orange line). Panels (d), (e), and (f) show the corresponding results for the nutrition estimation task.
![20981754463201_ pic](https://github.com/user-attachments/assets/ab489f0b-17c8-4af4-9100-e14de8b7ab79)


## Results
![20941754462912_ pic](https://github.com/user-attachments/assets/f43f0120-bfb7-4988-b619-3646dd1e189c)
