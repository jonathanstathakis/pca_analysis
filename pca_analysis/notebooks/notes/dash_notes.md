# Dash Notes

## Layout

https://dash.plotly.com/layout

- Apps are composed of two structural components - the *layout* and the *interactivity*
- the layout consists of a tree of *components*. For example there is a component available for every type of HTML tag. The `Dash.layout` attribute is set equal to this defined tree.
- Dash uses *React* for interactive components.
- dash has a *declarative* philosophy embodied through keyword argument values of components through which app design is controlled.
- for *hot-reloading* run app with `app.run(debug=True)`. This appears to be the default.
- Basic app file structure an app as follows:
    1. Create a file to be run as a script with `app.run` as the `main`
    2. create the app object in the global scope.
    3. add components to the app object.
    - presumably you can import the app object.
- HTML styles are supplied as dictionaries whose keys are in camelcase.
- `dash.dcc.Graph` is used to render visualisations.
- Markdown can be rendered through `dcc.Markdown` class.
- Selections can be made through `dcc` `Dropdown`, `Dropdown(multi=True)`, `RadioItems`, `Checklist`, `Slider` etc.

## Callbacks

https://dash.plotly.com/basic-callbacks

- User input can be taken through `dcc.Input`
- the `@callback` decorator wraps a function that returns the result of the callback. Taking Input and Output objects associated with html Divs, it watches the input value, and if it changes, runs the wrapped function, providing the output value. The callback and the Div are associated through the `component_id` arg, which must be equal to link the two.
- to connect the component inputs to the function parameters the `DashDependency.component_id` must match the function parameters name.
- The wrapped function should describe the output of the callback.
- This is called *reactive programming*
- case study - Selecting dataset groups through slider then vizzing:
    - Properties of objects such as `dcc.Slider` can be connected to callbacks. In this case, the slider has a `value` property which contains its current selected value.
    - The callback then gets the slider value as input then generates a plot with the data filtered by `value`.
- callbacks can watch multiple inputs at the same time.
- Callbacks can be used to return multiple values (tuples) which is useful in the case of a computationally intensive operation that is used by multiple functions downtrack. This can however create dependency issues.
- callbacks can be chained if one depends on the value of another.
- *state* can be managed in order to allow form-like inputs where multiple inputs are entered before a callback is activated. This is done by associating the callback `Input` with a value that doesnt change with the others, and hte other inputs are instead provided as `State` in the callback call. A classic example is to watch the `n_clicks` property of a `html.Button` which increments when the user clicks the button. The other inputs are passed as they are changed but the function isnt executed until the button is pressed (see the example).
- we can use the component objects direclty in the callback call in order to make things more pythonic and allow for tooling to catch errors and navigate the toolbase. Furthermore we can use walrus operators in the layout definition to define the component objects in place (see example).

## Interactive Visualisations

https://dash.plotly.com/interactive-graphing

## Multipage

- Use dcc.Store to share data between pages
- background on pages^[https://www.dash-extensions.com/sections/pages]: 
    - pages work by rendering the layout of the page whenever the url changes.
    - however this means that components are not shared between pages.
    - This is good in terms of scalability but bad in terms of page loading, depends on your needs.
    - 