# Attention Is All you Need

This Project will look at the seminal paper on transformer architecture, titled "Attention Is All you Need" by Vaswani et al. <br/> The Model architecture for the Transformer, formally presented in the paper mentioned above, is shown in Figure 1.

<img src="https://github.com/user-attachments/assets/a882c2de-a84b-4d5e-8d20-1e9f22a2240e" alt="model" width="300"/>

#### Figure 1: The Transformer - model architecture

The logic of the code follows the architecture presented in figure 1. Each step in the architecture is reproduced in the code as a class. These classes are called and stitched together in a way to mimic the above.



<img src="https://github.com/user-attachments/assets/77ead086-0656-45fd-9fbe-d8e83a86d50a" alt="pe" width="350"/>

#### Positional encoding (slightly different from our implementation)


<img src = "https://github.com/user-attachments/assets/b57bfaab-7c29-49fd-b9b5-3aa5f17b0c8e" alt = "pe2" width = "300"/>

#### Our implementation of the positional encoding equation in the code

<img src = "https://github.com/user-attachments/assets/9fed4eb3-287d-4536-9706-c1b60e0812b5" alt = "pe2" width = "100"/>

#### LayerNorm Equation

![image](https://github.com/user-attachments/assets/7954de65-e7bd-451e-99ed-46d6cebee200)
#### Dimesnions of parameters when going through Multi-HEaded attention
