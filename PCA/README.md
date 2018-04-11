# Task to government yields PCA

In the file you can find yield data from August 2000 until July 2010 of Swiss government bonds for times to maturities 2, 3, 4, 5, 7, 10, 20 and 30 years.

Answer the following 2 questions:

a) Perform a PCA on the monthly changes of the yield curve and report how much variance is explained by the first two principal components. Express your answer in percentage points and round to 2 decimal places.

b) Suppose in July 2010 you have a long position in a bond portfolio with the following cash flows:

|Year      |2   |3   |4    |5   |
|----------|----|----|-----|----|
|Cash Flows|80  |70  |150  |40  |

Approximate the price change of this portfolio over a time interval Δt to a first order as

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;V\approx&space;\frac{\partial&space;V}{\partial&space;t}\Delta&space;t&space;&plus;\sum_{i=1}^5\frac{\partial&space;V}{\partial&space;y_i}\Delta&space;y_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;V\approx&space;\frac{\partial&space;V}{\partial&space;t}\Delta&space;t&space;&plus;\sum_{i=1}^5\frac{\partial&space;V}{\partial&space;y_i}\Delta&space;y_i" title="\Delta V\approx \frac{\partial V}{\partial t}\Delta t +\sum_{i=1}^5\frac{\partial V}{\partial y_i}\Delta y_i" /></a>

where ∂V/∂t and ∂V/∂yi denote the partial derivatives of the portfolio value with respect to time and the i-year yield, respectively, and Δyi is the change of the i-year yield over the time interval Δt. Assume furthermore that the monthly change of the yield curve is described by the first two principal components:

<a href="https://www.codecogs.com/eqnedit.php?latex=\Delta&space;y_i\approx&space;\mu_i&plus;\sum_{j=1}^2Y_ja_{i,j}," target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Delta&space;y_i\approx&space;\mu_i&plus;\sum_{j=1}^2Y_ja_{i,j}," title="\Delta y_i\approx \mu_i+\sum_{j=1}^2Y_ja_{i,j}," /></a>

where Yj is the j-th principal components, a(i,j) is the loading of the monthly i-year yield change on the j-th principal component and μi is the average monthly change in the i-year yield.

Using the information from the PCA you performed in a), what is the empirical standard deviation of a monthly change of the portfolio value? Round your answer to 2 decimal places.
