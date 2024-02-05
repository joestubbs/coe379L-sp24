Linear Regression
=================

In this module, we introduce our first ML algorithm called Linear Regression. We also 
introduce the SciKit Learn Python library which provides an implementation of Linear 
Regression and many other ML algorithms. 

By the end of this module, students should be able to:

1. Understand the basics of the Linear Regression model and which ML problems it could
   potentially be applied to. 

2. Install and import the SciKit Learn Python package into a Python program, and use 
   SciKit Learn to implement a basic Linear Regression model.

Introduction
------------

In Linear Regression, we make the assumption that there is a linear relationship between the dependent 
and independent variables. To simplify the discussion, we'll assume for now that there are just 
two variables, one *independent* and one *dependent*. 

Recall from the previous lecture that our goal is to model (or predict) the dependent variable
from the independent variable. It is customary to use :math:`X` for the independent variable and 
:math:`Y` for the dependent variable. To say that there is a linear relationship between :math:`X` and :math:`Y` 
is to say that they are related by a linear equation.

We know from elementary algebra that a linear equation has the form 

.. math::

  Y - Y_1 = m(X- X_1)

and is uniquely determined by two points :math:`(X_1, Y_1)` and :math:`(X_2, Y_2)`. This is called the 
**point-slope form** of the linear equation. Note that by solving the left-hand side of the equation for 
:math:`Y`, we can put the equation in **slope-intercept** form: 

.. math::

   Y = mX + B 

Consider the case of predicting the market value of a piece of real estate. We know in the real world,
the value of a property depends on a number of factors, but for simplicity, let us make the assumption that the 
value is determined by the square footage. Let us further assume that the relationship is linear. 

We can restate the remarks above in this context as follows: Given the square footage and value of two properties, 
we can uniquely determine the linear equation relating square footage and value. Here are two properties in the 
Austin area that were recently listed on the MLS: 

* Property 1: 1,007 square feet; $320,000
* Property 2: 2,202 square feet; $561,000

We can simplify the data slightly be dividing by 1,000. 

Therefore, we can think of these properties as corresponding to the 
points :math:`(1, 320)` and :math:`(2.2, 561)` which leads to the system of equations:

.. math::

  Y - 320 = m(X- 1)

  Y - 561 = m(X- 2.2)

and then to the formula :math:`Y = 200.83(X - 1) + 320` which we can visualize as follows:


.. figure:: ./images/line_two_points.png
    :width: 1000px
    :align: center

Congratulations! In some sense, this is our very first linear model. It models the value of a 
real estate property (the :math:`Y` variable) as a linear function of square footage (the :math:`X` variable).

Using this formula, we could predict the value of another property based on its square footage. Here are
some additional properties. How does our model perform?

* Property 3: 2,550 square feet; actual value: $590,000; predicted value: ?
* Property 4: 3,202 square feet; actual value: $910,000; predicted value: ?
* Property 4: 1,500 square feet; actual value: $1,120,000; predicted value: ?

*Solution:*

We plug the points into the equation :math:`Y = 200.83(X - 1) + 320` and compute :math:`Y`:

* Property 3: Predicted Value = :math:`200.83(2.5-1) + 320 = $621,245`
* Property 4: Predicted Value = :math:`200.83(3.2-1) + 320 = $761,826` 
* Property 5: Predicted Value = :math:`200.83(1.5-1) + 320 = $420,415`

If we add these additional data points to our plot, we see that our model did pretty well on Property 3, 
less good on Property 4, and was completely wrong about Property 5. 

.. figure:: ./images/line_additional_points.png
    :width: 1000px
    :align: center

