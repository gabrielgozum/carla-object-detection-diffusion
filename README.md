# Diffusion Models to Denoise Adversarial Images

In this work we develop a simple approach to detecting adversarial regions in an image using diffusion models. The process follows:
* Add Gaussian noise to an adversarial image
* Run this image through the denoising process of diffusion model fine-tuned on a given domain
  * Note: This cannot be a latent diffusion model, must be ran at full resolution
* Compute the difference between the denoised image and the adversarial image
* Threshold the difference to generate a binary mask
* Run a BFS on the mask, then pass through to the object detector

The idea being the fine-tuned diffusion model is unable recognize the adversarial perturbations to the image; therefore, it is unable to faithfully reconstruct those regions. Taking the difference between the two images effectively pinpoints adversarial regions.


<img title="Adversarial CARLA Denoising Diffusion Workflow" alt="Adversarial CARLA Denoising Diffusion Workflow" src="/images/adversarial_carla_workflow.png">
