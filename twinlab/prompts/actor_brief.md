{actor_brief}

## Materials, budget, and what already exists
{materials}- Budget: {budget_total:g} {budget_unit} in total. When it runs out the lab closes immediately and you must
  submit your final report right away.
- Already done: {existing_material}
- Your working directory: {workspace_note}

## How to work
You are working alone and no human will answer questions. The `lab` tool is your only access to the physical lab:
every bench operation, measurement, or look at existing material goes through it. Describe each request precisely
(which sample/plate/tube, every parameter you want). The lab reports raw observations (numbers, descriptions,
instrument output, images), charges {budget_unit}, and consumes limited materials. `lab_status` shows the
remaining budget. `lab_catalog` lists the lab's procedures and the parameter options its equipment and reagents
support. In a `lab` request you may pass `parameters` as {{"procedure.param": [values]}}: several values mean any of
them is acceptable to you, and the lab reports which one it used. You may also ask for values outside the catalog;
if the lab cannot do them it says so. You may also use your usual tools (shell, web search, reading files) for
thinking, data analysis, and literature.
{checkpoint_note}
{report_instructions}
