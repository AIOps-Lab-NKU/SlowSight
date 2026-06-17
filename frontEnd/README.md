# Project Structure

## build
This folder contains a Node.js script that builds the Vue project based on different conditions and optionally starts a static server to preview the built project.

## public
This folder stores image information required by the front-end page tags.

## src
This directory contains the core part of the project, including functional pages, components, language switching, and routing.
### component
This folder stores information about the navigation menu on the right side of the page and the path display component at the top. You can modify the styles and data within this folder.
### lang
The project currently supports two languages. To add a new language, you need to add a new language file in the `lang` folder and modify the language selection function in `src/layout/components/Navbar.vue`.
### router
This file contains the routing configuration for the Vue project. You can set parameters here to associate with the navigation menu.
### views
This directory stores the code for various pages in the Vue project. The `line` folder contains the code for line chart operations, the `scatter` folder contains the code for scatter plot operations, and the `table` folder contains the code for the data management page shared by two pages. The `login` page is currently not included in the project requirements but can be added if needed in the future.