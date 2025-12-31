import numpy as np
import cv2
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
except ImportError:
    print("pydensecrf not found. DenseCRF will be skipped.")
    dcrf = None

def apply_morphology(mask, kernel_size=3):
    """
    Apply Morphological Closing to fill gaps.
    mask: Binary mask (0 or 1)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Closing: Dilation -> Erosion
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return closed

def apply_binary_dense_crf(prob_map, image, sxy=50, srgb=5, compat=10, n_iters=5):
    """
    Apply DenseCRF to a single channel probability map (Binary Classification).
    
    prob_map: (H, W) float [0, 1] - Probability of foreground
    image: (H, W, 3) uint8 - Original Image
    
    Returns:
    refined_prob: (H, W) float [0, 1]
    """
    if dcrf is None:
        return prob_map

    H, W = prob_map.shape
    
    # Create unary potential
    # 2 classes: Background (0), Foreground (1)
    # prob_map is P(Foreground)
    # We need P(Background) = 1 - P(Foreground)
    
    # Clip probabilities to avoid log(0)
    prob_map = np.clip(prob_map, 1e-6, 1.0 - 1e-6)
    
    # Stack to (2, H, W) -> [Background, Foreground]
    stacked_probs = np.stack([1.0 - prob_map, prob_map], axis=0)
    
    # Unary from softmax/probabilities
    # Input to UNARY should be negative log probabilities?
    # pydensecrf expects (N_LABELS, N_PIXELS)
    # U = -np.log(stacked_probs)
    # U = U.reshape(2, -1).astype(np.float32)
    # Actually unary_from_softmax takes softmax probs and returns suitable unary
    
    U = unary_from_softmax(stacked_probs) # (2, H*W)

    # Create CRF
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(U)

    # Pairwise Gaussian (Spatial smoothness)
    d.addPairwiseGaussian(sxy=3, compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Pairwise Bilateral (Appearance - edges)
    # sxy: spatial std (larger = longer range)
    # srgb: color std (smaller = sensitive to color diff)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgb=np.ascontiguousarray(image), compat=compat,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Inference
    Q = d.inference(n_iters)
    
    # Q is (2, H*W) probabilities
    res = np.array(Q).reshape((2, H, W))
    
    # Return Foreground probability
    return res[1]

def apply_dense_crf_multilabel(probs, image):
    """
    Apply CRF to each channel independently (since classes are not mutually exclusive).
    probs: (C, H, W)
    image: (H, W, 3)
    """
    C, H, W = probs.shape
    refined = np.zeros_like(probs)
    
    for c in range(C):
        refined[c] = apply_binary_dense_crf(probs[c], image)
        
    return refined
