# Edge detection

**Edges** are very important in human visual systems to distinguish objects

Edge can be defined as Place of rapid change in intensity

#### **Edges are caused by discontinuity of:**

- Surface color/ appearance
- Surface normal
- Depth
- Illumination

#### **Why Edges?**

- Group pixels into objects or parts
- To track important features (*corners, curves, lines*)
- Cues for 3D shape
- Guiding interactive image editing

### **Propoerties:**

- **Detection**
  - Find all real edge and ignore noise
- **Localization**
  - Detect edge close to true edge
- **Single response**
  - Return only one point for each true edge point

----

## **Image Gradients**

Gradient points the direction of the most rapid change in the intensity.

**Gradient of an image is given by**:
$$
\nabla f = [\frac{\partial f}{\partial x} , \frac{\partial f}{\partial y}]
$$

**Gradient direction (Orientation of the edge normal) is given by**:
$$
\theta = tan ^{-1} [\frac{\partial f}{\partial x} / \frac{\partial f}{\partial y}]
$$

**Edge strength (Magnitude) is given by:**
$$
|| \nabla f || = \sqrt{(\frac{\partial f}{\partial x})^2 + (\frac{\partial f}{\partial y})^2}
$$

We can approximate the partial derivative and one such approximation gives us Sobel Filter, which is widely used of edge detection, which is defined as:
$$
\begin{bmatrix}
-1 & 0 & 1\\
-2 & 0 & 2\\
-1 & 0 & 1\\
\end{bmatrix}
$$
$$Sobel \space Filter$$

**Few other first order derivative filters are**:

$$
\begin{bmatrix}
-1 & 0 & 1\\
-1 & 0 & 1\\
-1 & 0 & 1\\
\end{bmatrix}
$$
$$Prewitt \space filter$$

$$
\begin{bmatrix}
0 & 1\\
-1 & 0\\
\end{bmatrix}
$$
$$Roberts \space filter$$

## **Non Maxima Supression**

Used to thin out the edges, edges calculated using image gradient will output an image with thick edges, so in order to get fine thin edge we use non maximum supression

Non maxima supression is accomplished by checking if pixel is local maximum along the gradient

## **Hystersis Thresholding**

This technique uses two thresholds, namely high and low

- High: Strat the edge
- Low: Continue the edge

If the gradient at pixel is

- High(>): **Edge pixel**
- Low(<): Non edge pixel
- High & low(<=>): Edge pixel iff it is connected to an 'edge pixel' directly or via pixels between low and high

----

## **Canny Edge Detection:**

- Found by John Canny in 1986

### **Algorithm:**

1. Filter image with derivative of guassian
2. Find magnitude & orientation  of gradient
3. Non maximum supression
4. Linking & Thersholding (**Hystersis**):

- Define two thresholds: **low** & **high**
  - High: Strat the edge
  - Low: Continue the edge

Workflow of the Algorithm:

#### **Original --> Smoothed --> Gradient Magnitudes --> edge after non maximum supression --> Double thresholding --> Edge tracking by hystersis --> Final Output**

----

### **References**

1. <https://nptel.ac.in/courses/106106224>
2. <https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html>
3. <https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MORSE/edges.pdf>
4. <https://cse442-17f.github.io/Sobel-Laplacian-and-Canny-Edge-Detection-Algorithms/>
5. <https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123>
6. <http://www.adeveloperdiary.com/data-science/computer-vision/implement-canny-edge-detector-using-python-from-scratch/>
7. <https://github.com/StefanPitur/Edge-detection---Canny-detector>
8. <https://gist.github.com/FienSoP/03ed9b0eab196dde7b66f452725a42ac>
