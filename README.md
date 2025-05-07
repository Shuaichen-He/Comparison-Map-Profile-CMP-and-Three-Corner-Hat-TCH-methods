# TCH-CMP

## Uncertainty Analysis and Consistency Assessment of Evaporation Data in the Greater Mekong Subregion -- Journal Of Hydrometeorology Accepted

- In our paper, we evaluated the consistency and relative uncertainty of four evaporation datasets (ERA5-Land, FLUXCOM, GLDAS, GLEAM) in the Greater Mekong Subregion (GMS) using Comparison Map Profile (CMP) and Three Corner Hat (TCH) methods. We have made our Python code publicly available to assist other researchers.

- The *.npy data uploaded with the *.py files are examples, where m*.npy represents the multi-year averages of the four datasets in the GMS region, which can be used to calculate spatial similarity using the CMP method. The CMP method has two types based on different metrics calculated in the window: Distance and Cross-Correlation (CC), with the CC method being used in the study.

- The files named directly after the four datasets contain monthly average evaporation values, used for TCH calculation. TCH has multiple constraint condition methods for solving, with two types included in the code. The study used the H constraint condition for solving. For specific meanings, please refer to the reference literature or the data and methods section of the paper.

*The Three Corner Hat (TCH) method, proposed by Tavella and Premoli (1993).*

*Comparison of Map Profile (CMP) method was proposed by Gaucherel et al. (2007) to quantify the magnitude and distribution area of differences between image data.*

*Premoli, A., and P. Tavella, 1993: A revisited three-cornered hat method for estimating frequency standard instability. IEEE Trans. Instrum. Meas., 42, 7–13, https://doi.org/10.1109/19.206671.*

*Gaucherel, C., S. Alleaume, and C. Hély, 2007: The Comparison Map Profile method: a strategy for multiscale comparison of quantitative and qualitative images. 2007,.*
