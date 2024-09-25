import re
import numpy as np
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Dict, Union

from env4v import R2RNavBatch
from argparse import Namespace
from agent_base import BaseAgent

from langchain.agents.agent import AgentExecutor, AgentAction, AgentOutputParser
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.chains import LLMChain
from langchain.utils.input import get_colored_text
from LLMs.openai4v import OpenAIChat4v
from langchain.prompts import PromptTemplate
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseMessage,
    BaseOutputParser,
    OutputParserException
)
from langchain.base_language import BaseLanguageModel

from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from prompt.planner_prompt import (
    PLANNER_PROMPT,
    MAKE_ACTION_TOOL_NAME,
    MAKE_ACTION_TOOL_DESCRIPTION,
    BACK_TRACE_TOOL_NAME,
    BACK_TRACE_TOOL_DESCRIPTION,
    VLN_GPT4v_PROMPT,
)

FINAL_ANSWER_ACTION = "Final Answer:"
EXCEPTION_TOOL_NAME = "_Exception"
MAX_SCRATCHPAD_LENGTH = 7000

MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)

PROMPT_HISTORY="Thought: I should start navigation according to the instruction, {agent_scratchpad}"

class LLMChain4v(LLMChain):

    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager = None,
    ):
        """Prepare prompts from inputs from get_full_inputs of the agent."""
        stop = None
        if "stop" in input_list[0]:
            stop = input_list[0]["stop"]
        prompts = []
        prompt_infos = {}
        for inputs in input_list:
            selected_inputs = {}
            for k in inputs.keys():
                if k == "init_observation":
                    loc_prompt = []
                    loc_prompt.append(('text', '\nInitial Observation: \n'))
                    if isinstance(inputs["init_observation"], str):
                        loc_prompt.append(('text', inputs["init_observation"]))
                    else:
                        for item in inputs["init_observation"]:
                            if item[0] == 'text':
                                loc_prompt.append(('text', item[1]))
                            elif item[0] == 'vision':
                                loc_prompt.append(('vision', item[1]))
                            elif item[0] == 'text_obs':
                                continue
                            else:
                                import ipdb;ipdb.set_trace() # breakpoint 82  
                                print()
                    prompt_infos[k] = loc_prompt
                elif k in self.prompt.input_variables:
                    if k == 'agent_scratchpad':
                        selected_inputs[k] = ""
                        prompt_infos[k]  = PROMPT_HISTORY.format(agent_scratchpad=inputs[k])
                    else:
                        selected_inputs[k] = inputs[k]
                elif k == 'stop':
                    continue
                elif k == 'observation':
                    prompt_infos[k] = inputs[k]
                else:
                    import ipdb;ipdb.set_trace() # breakpoint 79
                    print()

            prompt = self.prompt.format_prompt(**selected_inputs)
            _colored_text = get_colored_text(prompt.to_string(), "green")
            _text = "Prompt after formatting:\n" + _colored_text
            if run_manager:
                run_manager.on_text(_text, end="\n", verbose=self.verbose)
            if "stop" in inputs and inputs["stop"] != stop:
                raise ValueError(
                    "If `stop` is present in any inputs, should be present in all."
                )
            prompts.append(('text', prompt.text))
            if 'init_observation' in prompt_infos.keys():
                if len(prompt_infos['init_observation']) == 2:
                    obs_text = prompt_infos['init_observation'][1][1]
                    prompts[0] = ('text', prompts[0][1] + f"Initial Observation: {obs_text}. \n")
                    prompts[0] = ('text', prompts[0][1] + prompt_infos['agent_scratchpad'])
                else:
                    prompts.extend(prompt_infos['init_observation'])

            if inputs['agent_scratchpad'] == '':
                prompts.append(('text', "Thought: I should start navigation according to the instruction, "))

            if 'observation' in prompt_infos.keys():
                if prompt_infos['observation'][0][0] not in ['text', 'vision']:
                    import ipdb;ipdb.set_trace() # breakpoint 112
                prompts.extend(prompt_infos['observation'])
                prompts.append(("text", "\nThought: "))

        return prompts, stop


class NavGPTOutputParser(AgentOutputParser):
    """MRKL Output parser for the chat agent."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*\"?([a-fA-F0-9]{32})\"?"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if action_match:
            if includes_answer:
                raise OutputParserException(
                    f"{FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE}: {text}"
                )
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        elif includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "mrkl-NavGPT"

class VLNAgent(ZeroShotAgent):

    history: Optional[List[str]] = None 

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[str, List[BaseMessage]]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts = ""
        nav_step = 1
        for i, (action, observation) in enumerate(intermediate_steps):
            thoughts += action.log
            if (i == len(intermediate_steps) - 1) or (action.tool != MAKE_ACTION_TOOL_NAME): 
                # current view
                # thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
                continue
            else:
                thoughts += f"\n{self.observation_prefix}{self.history[nav_step]}\n{self.llm_prefix}"
                nav_step += 1
        thoughts = thoughts.replace('Thought:Thought:', 'Thought:')
        return thoughts

    def get_full_inputs(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Create the full inputs for the LLMChain from intermediate steps."""
        if len(intermediate_steps) > 0:
            # NOTE: current view
            # assert observation[-1][0] == 'text_obs'
            # observation = observation[-1][1]
            observation = intermediate_steps[-1][1][:-1]

            # NOTE: Other Views
            thoughts = self._construct_scratchpad(intermediate_steps)[-MAX_SCRATCHPAD_LENGTH:]
            if observation[0][0] not in ['text', 'vision']:
                thoughts += f"\n{observation}\n"
                print(f'\033[1;31m [Warning]\033[0m {thoughts}')
                observation = None
        else:
            thoughts = ""
            observation = None

        new_inputs = {"agent_scratchpad": thoughts, "stop": self._stop}
        if observation is not None:
            new_inputs.update({"observation": observation})

        if len(intermediate_steps) == 0:
            full_inputs = {**kwargs, **new_inputs}
        else:
            kwargs["init_observation"] = self.history[0]
            full_inputs = {**kwargs, **new_inputs}
        return full_inputs
 

class NavAgent4v(BaseAgent):
    def __init__(
            self, 
            env: R2RNavBatch, 
            config: Namespace):
        """
        Initialize the LLM Navigation Agent.

        Args:
            env: The Matterport3D environment.
            config: The configuration.
        """
        super().__init__(env)
        self.config = config
        self.observation_prefix = "Observation: "

        if config.llm_model_name.split('-')[0] == 'gpt':
            self.llm = OpenAIChat4v(model=config.llm_model_name, max_tokens=900, temperature=config.temperature)
        else:
            raise ValueError(f"{config.llm_model_name} not support")

        self.output_parser = NavGPTOutputParser()
        self.agent_executor = self.create_vln_agent()

        plan_prompt = PromptTemplate(
            template=PLANNER_PROMPT,
            input_variables=["instruction"],
        )
        self.plan_chain = LLMChain4v(llm=self.llm, prompt=plan_prompt)
    
    def parse_action(self, llm_output: str) -> Tuple[str, str]:
        regex = r"(.*?)Final Answer:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")

        thought = match.group(1).strip()
        action = match.group(2).strip(" ").strip('"').strip("'")

        return thought, action
 
    def get_history(self, obs: dict, angle: str) -> str:
        '''Return the history of actions taken.'''
        history = f'{angle}\nCurrent viewpoint "{obs["viewpoint"]}": Scene from the viewpoint is a {obs["obs_summary"]}'
        return history

    def modify_heading_angles(self, heading_angle, observation_list, candidate_dict, object_list, observation_list_text):
        # Function to normalize an angle to the range of -180 to 180
        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle <= -180:
                angle += 360
            return angle
        
        def angle_to_left_right(angle):
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"
        
        # Define the directions
        directions = ['Front', 'Front Right', 'Right', 'Rear Right', 'Rear', 'Rear Left', 'Left', 'Front Left']

        # Calculate the range of heading angles belonging to each direction
        range_idx = int((heading_angle - 22.5) // 45) + 1
        obs_idx = [(i + range_idx) % 8 for i in range(8)]
        
        # Initialize a dictionary to store the candidate viewpoints for each direction
        candidate_range = {}
        if not self.config.use_navigable:
            for viewpoint_id, viewpoint_data in candidate_dict.items():
                viewpoint_heading = np.rad2deg(viewpoint_data['heading'])
                vp_range_idx = int((viewpoint_heading - 22.5) // 45) + 1
                rel_viewpoint_heading = viewpoint_heading - heading_angle
                rel_viewpoint_heading = normalize_angle(rel_viewpoint_heading)
                rel_viewpoint_heading = angle_to_left_right(rel_viewpoint_heading)
                vp_description = rel_viewpoint_heading + f', {viewpoint_data["distance"]:.2f}m'
                # rel_range_idx = (vp_range_idx - range_idx) % 8
                candidate_range.setdefault(vp_range_idx, {}).update({viewpoint_id: vp_description})

        # Calculate the relative angle ranges based on the heading angle
        angle_ranges = [(angle - 22.5 - heading_angle, angle + 22.5 - heading_angle) for angle in range(0, 360, 45)]
        
        # Initialize an empty list to store the formatted strings
        # formatted_strings = []
        formatted_msgs = []
        text_obs = []
        
        # Iterate through the directions, angle ranges, and observation strings
        for obs in observation_list:
            formatted_msgs.append(('vision', obs))

        for ii, (direction, idx) in enumerate(zip(directions, obs_idx)):
            # Calculate the relative angles and normalize them
            rel_angle1 = normalize_angle(angle_ranges[idx][0])
            rel_angle2 = normalize_angle(angle_ranges[idx][1])

            # Convert the angles to "left n" or "right n"
            left_right1 = angle_to_left_right(rel_angle1)
            left_right2 = angle_to_left_right(rel_angle2)
            
            # Create the formatted string
            formatted_string = ""
            text_obs.append(f"{direction}, range ({left_right1} to {left_right2}): \n'{observation_list_text[idx]}'")
            # formatted_string = f"{direction}, range ({left_right1} to {left_right2}): \n'{observation_list[idx]}'"
            formatted_string = f"{direction} (Photo {ii}), range ({left_right1} to {left_right2}): \n'{observation_list_text[idx]}'"
            # formatted_msgs.append(('text', f"{direction} (Photo {ii}), range ({left_right1} to {left_right2}): \n"))
            
            # Add the objects to the formatted string
            object_dict = {}
            if len(object_list[idx]) > 0:
                object = object_list[idx]
                for obj, obj_data in object.items():
                    rel_obj_heading = obj_data['heading'] - heading_angle
                    rel_obj_heading = normalize_angle(rel_obj_heading)
                    rel_obj_heading = angle_to_left_right(rel_obj_heading)
                    object_dict[obj] = f'{rel_obj_heading}, {obj_data["distance"]:.2f}m'
                formatted_string += f'\n{direction} (Photo {ii}) Objects in 3m: {object_dict}'
            else:
                formatted_string += f'\n{direction} (Photo {ii}) Objects in 3m: None'

            # Add the candidate viewpoints to the formatted string
            if candidate_range.get(idx):
                formatted_string += f'\n{direction} (Photo {ii}) Navigable Viewpoints:{candidate_range[idx]}'
            else:
                formatted_string += f'\n{direction} (Photo {ii}) Navigable Viewpoints: None'
            
            formatted_msgs.append(('text', formatted_string + "\n"))
            print(formatted_string)
   
            # # Add the formatted string to the list
            # formatted_strings.append(formatted_string)
        
        # Join the formatted strings into a single output string
        # output_string = '\n'.join(formatted_strings)
        # return output_string
        text_obs = '\n'.join(text_obs)
        formatted_msgs.append(('text_obs', text_obs))
        return formatted_msgs

    def init_trajecotry(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        # Record the navigation path
        self.traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': [],
        } for ob in obs]
        # Record the history of actions taken
        self.agent_executor.agent.history = [f'Navigation start, no actions taken yet.\nCurrent viewpoint "{obs[0]["viewpoint"]}": Scene from the viewpoint is a {obs[0]["obs_summary"]}']

    def _create_make_action_tool(
            self,
            llm: BaseLanguageModel,
    ) -> Tool:
        """Create a tool to make single action prediction in MP3D.

        The tool is invoked with the simulation environment and records the
        action taken by the agent.
        The tool interacts with the environment to obtain the current observation, 
        uses the LLM to predict the next action, and to summarize the previous trajectory
        into history.
        """

        # action_prompt = PromptTemplate(
        #     template=ACTION_PROMPT,
        #     input_variables=["action_plan", "observation", "history", "navigable_viewpoints"],
        # )
        # history_prompt = PromptTemplate(
        #     template=HISTORY_PROMPT,
        #     input_variables=["history", "previous_action", "observation"],
        # )
        
        # self.action_chain = LLMChain(llm=llm, prompt=action_prompt)
        # self.history_chain = LLMChain(llm=llm, prompt=history_prompt)

        def _make_action(*args, **kwargs) -> str:
            '''Make single step action in MatterSim.'''
            # Get current observation
            cur_obs = self.env._get_obs()[0]

            # Get current feature
            feature_text = cur_obs['obs']
            feature = cur_obs['obs_vis']
            heading = np.rad2deg(cur_obs['heading'])
            elevation = np.rad2deg(cur_obs['elevation'])
            objects = cur_obs['objects']
            orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
            navigable = cur_obs['candidate']

            if self.config.use_relative_angle:
                feature = self.modify_heading_angles(heading, feature, navigable, objects, feature_text)
            if self.config.use_navigable:
                import ipdb;ipdb.set_trace() # breakpoint 384
                # navigable = self.get_navigable_str(heading, elevation, navigable)
            
            if self.config.use_tool_chain:
                # Get current action plan
                # action_plan = self.cur_action_plan
                # Single step action
                import ipdb;ipdb.set_trace() # breakpoint 386
                # LLM_action_output = self.action_chain.run(
                #     action_plan = action_plan, 
                #     observation = feature, 
                #     history = self.agent_executor.agent.history[-1], 
                #     navigable_viewpoints = navigable
                # )
                # # Parse LLM output, action is the next viewpoint ID
                # thought, action = self.parse_action(LLM_action_output)
            else:
                action = args[0].strip(" ").strip('"').strip("'")

            # Make the action in Simulator
            if action not in self.env.env.sims[0].navigable_dict.keys():
                # Update history
                history = f'ViewpointID "{action}" is not valid, no action taken for the agent.'
                self.agent_executor.agent.history.append(history)
                if self.config.use_navigable:
                    import ipdb;ipdb.set_trace() # breakpoint 469
                    # return f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                else:
                    prompts = []
                    prompts.append(("text", f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n"))
                    prompts += feature
                    return prompts 
                    # f"\nViewpointID '{action}' is not valid, agent not moved. DO NOT fabricate nonexistent IDs. The navigable viewpoints you can choose from current viewpoints are: {[key for key in navigable.keys()]}.\n\tCurrent Viewpoint:\n{feature}"
            else:
                turned_angle, new_obs = self.make_equiv_action([action])

            # Update the current feature
            new_feature_text = new_obs['obs']
            new_feature = new_obs['obs_vis']
            new_feature_sum = new_obs['obs_summary']
            new_navigable = new_obs['candidate']
            new_objects = new_obs['objects']
            new_heading = np.rad2deg(new_obs['heading'])
            new_elevation = np.rad2deg(new_obs['elevation'])
            if self.config.use_relative_angle:
                new_feature = self.modify_heading_angles(new_heading, new_feature, new_navigable, new_objects, new_feature_text)
            new_orientation = f'\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}'
            if self.config.use_navigable:
                import ipdb;ipdb.set_trace() # breakpoint 510
                # new_navigable = self.get_navigable_str(new_heading, new_elevation, new_navigable)

            # Update history
            if self.config.use_history_chain:
                import ipdb;ipdb.set_trace() # breakpoint 425
                # history = self.history_chain.run(
                #     observation = new_feature_sum, 
                #     history = self.agent_executor.agent.history[-1], 
                #     previous_action = turned_angle
                # )
            else:
                history = self.get_history(new_obs, turned_angle)

            self.agent_executor.agent.history.append(history)
            # Record single step detail
            if self.config.use_tool_chain:
                import ipdb;ipdb.set_trace() # breakpoint 442
                # detail = {
                #     "viewpointID": action,
                #     "turned_angle": turned_angle,
                #     "acion_maker_thought": thought,
                #     "feature": new_feature,
                #     "history": self.agent_executor.agent.history[-1],
                # }
            else:
                detail = {
                    "viewpointID": action,
                    "turned_angle": turned_angle,
                    "feature": new_feature,
                    "history": self.agent_executor.agent.history[-1],
                }
            self.traj[0]['details'].append(detail)
            # Return LLM chain output as the observation of tool
            if self.config.use_tool_chain:
                import ipdb;ipdb.set_trace() # breakpoint 460
                # return f"\n\tAction_maker Thought:\n{thought}\n\tAction_maker Action:\n{turned_angle}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
            elif self.config.use_relative_angle:
                if self.config.use_navigable:
                    import ipdb;ipdb.set_trace() # breakpoint 510
                    # return f"\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                else:
                    prompts = []
                    prompts.append(("text", f"\n{self.observation_prefix}Current Viewpoint:\n"))
                    prompts += new_feature
                    return prompts 
                    # return f'\nCurrent Viewpoint "{action}":\n{new_feature}'
            else:
                import ipdb;ipdb.set_trace() # breakpoint 470
                # if self.config.use_navigable:
                #     return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}\n\tNavigable Viewpoints:\n{new_navigable}"
                # else:
                #     return f"\n\tCurrent Orientation:\n{new_orientation}\n\tCurrent Viewpoint:\n{new_feature}"
             

        return Tool(
            name=MAKE_ACTION_TOOL_NAME,
            func=_make_action,
            description=MAKE_ACTION_TOOL_DESCRIPTION,
        )

    def _create_back_trace_tool(
            self,
            llm: BaseLanguageModel,
    ) -> Tool:
        """Create a tool to back trace during navigation.

        The tool is invoked with the history of navigation trajectory.
        Using the LLM to find a viewpoint on the trajectory to back trace to.
        """
        # prompt = PromptTemplate(
        #     template=BACK_TRACE_PROMPT,
        #     input_variables=["action_plan", "history", "observation"],
        # )

        # chain = LLMChain(llm=llm, prompt=prompt)

        def _back_trace(*args, **kwargs) -> str:
            '''Back trace the action plan.'''
            cur_obs = self.env._get_obs()[0]

            # Get current feature
            feature_text = cur_obs['obs']
            feature = cur_obs['obs_vis']
            navigable = cur_obs['candidate']
            objects = cur_obs['objects']
            heading = np.rad2deg(cur_obs['heading'])
            elevation = np.rad2deg(cur_obs['elevation'])
            orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
            if self.config.use_relative_angle:
                feature = self.modify_heading_angles(heading, feature, navigable, objects, feature_text)
            if self.config.use_navigable:
                raise ValueError("Setting not support")

            if self.config.use_tool_chain:
                raise ValueError("Setting not support")
            else:
                action = args[0].strip(" ").strip('"').strip("'")

            # Make the action in Simulator
            if action not in self.env.env.sims[0].navigable_dict.keys():
                if self.config.use_navigable:
                    raise ValueError("Setting not support")
                else:
                    prompts = []
                    prompts.append(("text", f"\nViewpointID '{action}' is not valid. DO NOT fabricate nonexistent IDs.\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n"))
                    prompts += feature
                    return prompts 
            else:
                _, new_obs = self.make_equiv_action([action])
            
            # Update the current feature
            new_feature_text = new_obs['obs']
            new_feature = new_obs['obs_vis']
            new_navigable = new_obs['candidate']
            new_objects = new_obs['objects']
            new_heading = np.rad2deg(new_obs['heading'])
            new_elevation = np.rad2deg(new_obs['elevation'])
            new_orientation = f'\nheading: {new_heading:.2f}, elevation: {new_elevation:.2f}'
            if self.config.use_relative_angle:
                new_feature = self.modify_heading_angles(new_heading, new_feature, new_navigable, new_objects, new_feature_text)
            if self.config.use_navigable:
                raise ValueError("Setting not support")

            # Update history
            history = self.get_history(new_obs, 'Seems going in a wrong way, back trace to a previous point.')
            self.agent_executor.agent.history.append(history)
            # Record single step detail
            if self.config.use_tool_chain:
                raise ValueError("Setting not support")
            elif self.config.use_relative_angle:
                if self.config.use_navigable:
                    raise ValueError("Setting not support")
                else:
                    prompts = []
                    prompts.append(("text", f"\n{self.observation_prefix}\nCurrent Viewpoint:{action}\n"))
                    prompts += new_feature
                    return prompts 
            else:
                raise ValueError("Setting not support")

        return Tool(
            name=BACK_TRACE_TOOL_NAME,
            func=_back_trace,
            description=BACK_TRACE_TOOL_DESCRIPTION,
        )

    def create_vln_agent(
        self,
    ) -> AgentExecutor:
        """Instantiate API planner and controller for a given trajectory.

        We use a top-level "orchestrator" agent to invoke the planner and controller,
        rather than a top-level planner
        that invokes a controller with its plan. This is to keep the planner simple.
        """

        self.action_maker = self._create_make_action_tool(self.llm)
        self.back_tracer = self._create_back_trace_tool(self.llm)

        tools = [
            self.action_maker,
            self.back_tracer
        ]
        
        if self.config.use_tool_chain:
            import ipdb;ipdb.set_trace() # breakpoint 583
            # prompt = PromptTemplate(
            #     template=VLN_ORCHESTRATOR_PROMPT,
            #     input_variables=["action_plan", "init_observation", "observation", "agent_scratchpad"],
            #     partial_variables={
            #         "tool_names": ", ".join([tool.name for tool in tools]),
            #         "tool_descriptions": "\n".join(
            #             [f"{tool.name}: {tool.description}" for tool in tools]
            #         ),
            #     },
            # )
        elif self.config.use_single_action:
            tools = [self.action_maker]
            prompt = PromptTemplate(
                template=VLN_GPT4v_PROMPT,
                # input_variables=["action_plan", "init_observation", "agent_scratchpad"],
                input_variables=["action_plan", "agent_scratchpad"], 
                partial_variables={
                    "tool_names": ", ".join([tool.name for tool in tools]),
                    "tool_descriptions": "\n".join(
                        [f"{tool.name}: {tool.description}" for tool in tools]
                    ),
                },
            )
            self.gpt4v_prompt = prompt
        else:
            import ipdb;ipdb.set_trace() # breakpoint 607
            # prompt = PromptTemplate(
            #     template=VLN_ORCHESTRATOR_PROMPT,
            #     input_variables=["action_plan", "init_observation", "agent_scratchpad"],
            #     partial_variables={
            #         "tool_names": ", ".join([tool.name for tool in tools]),
            #         "tool_descriptions": "\n".join(
            #             [f"{tool.name}: {tool.description}" for tool in tools]
            #         ),
            #     },
            # )

        agent = VLNAgent(
            llm_chain=LLMChain4v(llm=self.llm, prompt=prompt),
            allowed_tools=[tool.name for tool in tools],
            output_parser = self.output_parser,
        )

        return AgentExecutor.from_agent_and_tools(
            agent=agent, 
            tools=tools, 
            verbose=False, 
            handle_parsing_errors = True,
            return_intermediate_steps=True,
            max_iterations=self.config.max_iterations,
        )
    
    def make_equiv_action(self, actions: List[str]) -> str:
        """
        Interface between Panoramic view and Egocentric view
        Take in the next viewpoint ID and move the agent to that viewpoint
        return the turned angle and new observation
        """
        def normalize_angle(angle):
            while angle > 180:
                angle -= 360
            while angle <= -180:
                angle += 360
            return angle

        def angle_to_left_right(angle):
            return f"left {-angle:.2f}" if angle < 0 else f"right {angle:.2f}"
        
        # Get current agent facing angle
        cur_obs = self.env._get_obs()[0]
        cur_heading = np.rad2deg(cur_obs['heading'])
        # Make the action
        new_obs = self.env.step(actions)[0]
        new_heading = np.rad2deg(new_obs['heading'])
        # Record the trajectory
        self.traj[0]['path'].append(self.env.env.sims[0].gmap.bfs_shortest_path(cur_obs['viewpoint'], actions[0])[1:])
        # Calculate the turned angle
        turned_angle = new_heading - cur_heading
        # Generate action description
        cur_heading = angle_to_left_right(normalize_angle(cur_heading))
        new_heading = angle_to_left_right(normalize_angle(new_heading))
        action_description = f'Turn heading direction {turned_angle:.2f} degrees from {cur_heading} to {new_heading}.'
        return action_description, new_obs

    def rollout(self, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()

        # Initialize the trajectory
        self.init_trajecotry(obs)

        # Load the instruction
        instructions = [ob['instruction'] for ob in obs]
        if self.config.load_instruction:
            action_plans = instructions
        elif self.config.load_action_plan:
            import ipdb;ipdb.set_trace() # breakpoint 679
            # action_plans = [ob['action_plan'] for ob in obs]
        else:
            import ipdb;ipdb.set_trace() # breakpoint 682
            # action_plans = []
            # for instruction in instructions:
            #     action_plan = self.plan_chain.run(instruction = instruction)
            #     action_plans.append(action_plan)

        for i, init_ob in enumerate(obs):
            self.cur_action_plan = action_plans[i]
            # Take the first action
            if self.config.use_tool_chain:
                import ipdb;ipdb.set_trace() # breakpoint 689
                # first_obs = self.action_maker('')
                # input = {
                #     'action_plan': self.cur_action_plan,
                #     'init_observation': init_ob['obs_summary'],
                #     'observation': first_obs,
                # }
            else:
                # Get current feature
                feature_text = init_ob['obs']
                feature = init_ob['obs_vis']
                navigable = init_ob['candidate']
                objects = init_ob['objects']
                heading = np.rad2deg(init_ob['heading'])
                elevation = np.rad2deg(init_ob['elevation'])
                orientation = f'\nheading: {heading:.2f}, elevation: {elevation:.2f}'
                if self.config.use_relative_angle:
                    feature = self.modify_heading_angles(heading, feature, navigable, objects, feature_text)

                if self.config.use_navigable:
                    import ipdb;ipdb.set_trace() # breakpoint 707
                    # navigable = self.get_navigable_str(heading, elevation, navigable)

                if self.config.use_relative_angle:
                    if self.config.use_navigable:
                        # init_observation = f"\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                        import ipdb;ipdb.set_trace() # breakpoint 811
                    else:
                        # init_observation = f"\n\tCurrent Viewpoint:\n{feature}"
                        prompts = []
                        prompts.append(("text", f"\n\tCurrent Viewpoint:\n"))
                        prompts += feature
                        init_observation = prompts 
                        
                else:
                    if self.config.use_navigable:
                        # init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}\n\tNavigable Viewpoints:\n{navigable}"
                        import ipdb;ipdb.set_trace() # breakpoint 818
                    else:
                        # init_observation = f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n{feature}"
                        prompts = []
                        prompts.append(("text", f"\n\tCurrent Orientation:\n{orientation}\n\tCurrent Viewpoint:\n"))
                        prompts += feature
                        init_observation = prompts 

                input = {
                    'action_plan': self.cur_action_plan,
                    'init_observation': init_observation,
                }

            output = self.agent_executor(input)

            self.traj[i]['llm_output'] = output['output']
            self.traj[i]['action_plan'] = output['action_plan']
            # extract agent's thought from llm output
            intermediate_steps = output['intermediate_steps']
            self.traj[i]['llm_thought'] = []
            self.traj[i]['llm_observation'] = []
            for action, observation in intermediate_steps:
                thought = action.log
                self.traj[i]['llm_thought'].append(thought)
                self.traj[i]['llm_observation'].append(observation)

        return self.traj