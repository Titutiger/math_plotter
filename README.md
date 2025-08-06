# Disclamer!
While yes, you are permitted to use this at your will, I would really appreciate it if you would tag me / credit me.
Thanking you / you lot in advance!!

# Overview

## Format

>>> expr = lambda x: <input expression here> // for example: (x**2) + 2 * x + 5
>>> x, y = evaluate_expression(expr) // in here, loops and negative are 10 and False respectively by default.

Now, think of expr as f(x). var_x here is f(x)...
Making var_x = 5 as f(5).
And var_y is the output of f(x).

Now, for making the graph, there are 2 types:
Static (centered / uncentered)
Animated (centered / uncentered)
// Currently, these are 2 different functions, in later models, I will update it so that 'centered' is a bool param in the main function.

>>> Graphing.animate_graph(x, y) // for uncentered animated graph.
>>> Graphing.static_graph(x, y) // for uncentered animated graph.
// Please refer to the docstrings for centered ones.

## Purpose

When I was in school, it was hard to visulise graphs for many of the students. And when we did plot them, it was only for 2 or 5 points.
This was made in nostalgia such that one can actually see what is happenning!
