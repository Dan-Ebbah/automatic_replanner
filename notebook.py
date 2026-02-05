# %%
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple, Dict, Any, FrozenSet
from enum import Enum, auto

class ActionStatus(Enum):
    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass(frozen=True)
class Action:
    name: str
    agent_id: str
    preconditions: FrozenSet[str]
    positive_effects: FrozenSet[str]
    negative_effects: FrozenSet[str]
    capability: Optional[str] = None

    def _is_applicable(self, state: Set[str]) -> bool:
        return self.preconditions.issubset(state)

    def apply(self, state: Set[str]) -> Set[str]:
        return (state - self.negative_effects).union(self.positive_effects)


@dataclass
class ExecutionState:
    current_propositions: Set[str] = field(default_factory=set)
    action_statuses: Dict[str, ActionStatus] = field(default_factory=dict)
    agent_workflow_states: Dict[str, str] = field(default_factory=dict)
    goal: Set[str] = field(default_factory=set)

    def get_completed_actions(self) -> List[str]:
        return [
            name for name, status in self.action_statuses.items()
            if status == ActionStatus.COMPLETED
        ]

    def get_pending_actions(self) -> List[str]:
        return [
            name for name, status in self.action_statuses.items()
            if status in (ActionStatus.PENDING, ActionStatus.EXECUTING)
        ]

@dataclass
class AgentRegistry:
    agent_actions: Dict[str, List[Action]] = field(default_factory=dict)
    capability_providers: Dict[str, Set[str]] = field(default_factory=dict)
    available_agents: Set[str] = field(default_factory=set)

    def register_agent(self, agent_id: str, actions: List[Action]):
        self.agent_actions[agent_id] = actions
        self.available_agents.add(agent_id)

        for action in actions:
            if action.capability:
                if action.capability not in self.capability_providers:
                    self.capability_providers[action.capability] = set()
                self.capability_providers[action.capability].add(agent_id)

    def mark_unavailable(self, agent_id: str):
        self.available_agents.discard(agent_id)

    def mark_available(self, agent_id: str):
        if agent_id in self.available_agents:
            self.available_agents.add(agent_id)

    def get_available_actions(self) -> List[Action]:
        actions = []
        for agent_id in self.available_agents:
            actions.extend(self.agent_actions.get(agent_id, []))
        return actions

    def find_alternative_agents(self, failed_agent_id: str) -> Dict[str, Set[str]]:
        alternatives = {}

        failed_actions = self.agent_actions.get(failed_agent_id, [])
        failed_capabilities = {
            action.capability for action in failed_actions if action.capability
        }

        for capability in failed_capabilities:
            providers = self.capability_providers.get(capability, set())
            alternative_agents = providers & self.available_agents - {failed_agent_id}
            if alternative_agents:
                alternatives[capability] = alternative_agents
        return alternatives
# %%
## Graph Planning Engine
@dataclass
class PlanningGraph:
    @dataclass
    class Layer:
        propositions: Set[str] = field(default_factory=set)
        actions: Set[Action] = field(default_factory=set)
        proposition_mutexes: Set[FrozenSet[str]] = field(default_factory=set)
        action_mutex: Set[FrozenSet[Action]] = field(default_factory=set)

    initial_state: Set[str]
    goal: Set[str]
    available_actions: List[Action]
    layers: List[Layer] = field(default_factory=list)

    def build_graph(self, max_layers: int = 20) -> bool:
        initial_layer = self.Layer(propositions=self.initial_state.copy())
        self.layers = [initial_layer]

        for i in range(max_layers):
            current_props = self.layers[-1].propositions

            if self.goal.issubset(current_props):
                if not self._goal_has_mutex(self.layers[-1]):
                    return True

            next_actions = set()
            for action in self.available_actions:
                if action._is_applicable(current_props):
                    if not self._preconditions_mutex(action, self.layers[-1]):
                        next_actions.add(action)

            next_props = current_props.copy()
            for action in next_actions:
                next_props |= action.positive_effects

            action_mutexes = self._compute_action_mutexes(next_actions)
            prop_mutexes = self._compute_proposition_mutexes(next_props, next_actions, action_mutexes, self.layers[-1])

            new_layer = self.Layer(
                propositions=next_props,
                actions=next_actions,
                proposition_mutexes=prop_mutexes,
                action_mutex=action_mutexes
            )

            if (next_props == current_props and prop_mutexes == self.layers[-1].proposition_mutex):
                return False

            self.layers.append(new_layer)

        return False

    def _goal_has_mutex(self, layer: Layer) -> bool:
        goal_list = list(self.goal)
        for i, p1 in enumerate(goal_list):
            for p2 in goal_list[i + 1 :]:
                if frozenset({p1, p2}) in layer.proposition_mutexes:
                    return True

        return False

    def _preconditions_mutex(self, action: Action, layer: Layer) -> bool:
        preconds = list(action.preconditions)
        for i, p1 in enumerate(preconds):
            for p2 in preconds[i + 1 :]:
                if frozenset({p1, p2}) in layer.proposition_mutexes:
                    return True
        return False

    def _compute_action_mutexes(self, actions: Set[Action]) -> Set[FrozenSet[Action]]:
        mutexes = set()
        action_list = list(actions)
        for i, a1 in enumerate(action_list):
            for a2 in action_list[i + 1 :]:
                if (a1.positive_effects & a2.negative_effects or
                a2.positive_effects & a1.negative_effects):
                    mutexes.add(frozenset({a1, a2}))
                    continue

                if (a1.negative_effects & a2.preconditions or
                a2.negative_effects & a1.preconditions):
                    mutexes.add(frozenset({a1, a2}))
        return mutexes

    def _compute_proposition_mutexes(self, propositions: Set[str], actions: Set[Action], action_mutexes: Set[FrozenSet[Action]], previous_layer: Layer) -> Set[FrozenSet[str]]:
        mutexes = set()
        prop_list = list(propositions)

        achievers = {p: set() for p in propositions}
        for action in actions:
            for prop in action.positive_effects:
                if prop in achievers:
                    achievers[prop].add(action)

        for i, p1 in enumerate(prop_list):
            for p2 in prop_list[i + 1 :]:
                ways_to_achieve_p1 = achievers[p1]
                ways_to_achieve_p2 = achievers[p2]

                p1_can_persist = p1 in previous_layer.propositions
                p2_can_persist = p2 in previous_layer.propositions

                persist_mutex = frozenset({p1, p2}) in previous_layer.proposition_mutexes

                all_pairs_mutex = True

                for a1 in ways_to_achieve_p1:
                    for a2 in ways_to_achieve_p2:
                        if a1 == a2:
                            all_pairs_mutex = False
                            break
                        if frozenset({a1, a2}) not in action_mutexes:
                            all_pairs_mutex = False
                            break
                    if not all_pairs_mutex:
                        break

                if all_pairs_mutex:
                    if p1_can_persist and not persist_mutex:
                        for a2 in ways_to_achieve_p2:
                            if p1 not in a2.negative_effects:
                                all_pairs_mutex = False
                                break

                    if p2_can_persist and not persist_mutex:
                        for a1 in ways_to_achieve_p1:
                            if p2 not in a1.negative_effects:
                                all_pairs_mutex = False
                                break

                    if p1_can_persist and p2_can_persist and not persist_mutex:
                        all_pairs_mutex = False

                if all_pairs_mutex:
                    mutexes.add(frozenset({p1, p2}))

        return mutexes

    def extract_plan(self) -> Optional[List[Action]]:

        if not self.layers or not self.goal.issubset(self.layers[-1].propositions):
            return None

        return self._backward_search(len(self.layers) - 1, self.goal)

    def _backward_search(self, layer_index: int, subgoal: Set[str]) -> Optional[List[Action]]:
        if layer_index == 0:
            if subgoal.issubset(self.layers[0].propositions):
                return []

            return None

        layer = self.layers[layer_index]

        goal_achievers: Dict[str, List[Action]] = {}
        for goal in subgoal:
            achievers = []
            for action in layer.actions:
                if goal in action.positive_effects:
                    achievers.append(action)

            if goal in self.layers[layer_index - 1].propositions:
                achievers.append(None)

            goal_achievers[goal] = achievers

        selected = self._select_achievers(goal_achievers, layer)

        if selected is None:
            return None

        new_goals = set()
        for action in selected:
            if action is not None:
                new_goals |= action.preconditions

        for goal, achiever in zip(subgoal, selected):
            if achiever is None:
                new_goals.add(goal)

        sub_plan = self._backward_search(layer_index - 1, new_goals)

        if sub_plan is None:
            return None

        parallel_actions = {a for a in selected if a is not None}

        if parallel_actions:
            sub_plan.append(parallel_actions)

        return sub_plan

    def _select_achievers(self, goal_achievers: Dict[str, List[Optional[Action]]], layer: Layer) -> Optional[List[Optional[Action]]]:
        goals = list(goal_achievers.keys())
        selected = []

        for goal in goals:
            achievers = goal_achievers[goal]
            found = False

            for achiever in achievers:
                # Check mutex with already-selected achievers
                conflict = False
                for prev_achiever in selected:
                    if achiever is None or prev_achiever is None:
                        continue
                    if frozenset({achiever, prev_achiever}) in layer.action_mutex:
                        conflict = True
                        break

                if not conflict:
                    selected.append(achiever)
                    found = True
                    break

            if not found:
                return None

        return selected

# %%
@dataclass
class Plan:
    """
    Represents a composition plan: a sequence of parallel action sets.

    Each step in the plan contains actions that can be executed concurrently.
    The steps must be executed in order.
    """
    steps: List[Set[Action]] = field(default_factory=list)

    def get_all_actions(self) -> Set[Action]:
        """Get all actions in the plan regardless of step."""
        return {action for step in self.steps for action in step}

    def get_involved_agents(self) -> Set[str]:
        """Get IDs of all agents involved in this plan."""
        return {action.agent_id for action in self.get_all_actions()}

    def remove_agent_actions(self, agent_id: str) -> 'Plan':
        """Create a new plan with all actions from the specified agent removed."""
        new_steps = []
        for step in self.steps:
            filtered_step = {a for a in step if a.agent_id != agent_id}
            if filtered_step:
                new_steps.append(filtered_step)
        return Plan(steps=new_steps)


class SelfHealingController:
    """
    Orchestrates plan execution with automatic repair on agent failure.

    This is the heart of the self-healing system. It executes plans step by
    step, monitors for agent unavailability, and triggers replanning when
    failures are detected.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.current_plan: Optional[Plan] = None
        self.execution_state: Optional[ExecutionState] = None
        self.current_step_index: int = 0

    def create_initial_plan(
            self,
            initial_data: Set[str],
            goal: Set[str]
    ) -> Optional[Plan]:
        """
        Generate the initial composition plan using graph planning.

        This is analogous to what pycompose does in the paper, but we're
        working with agents instead of web services.
        """
        available_actions = self.registry.get_available_actions()

        graph = PlanningGraph(
            initial_state=initial_data,
            goal=goal,
            available_actions=available_actions
        )

        if not graph.build_graph():
            print("No plan exists with current agents and data.")
            return None

        plan_steps = graph.extract_plan()
        if plan_steps is None:
            print("Goal appears reachable but plan extraction failed.")
            return None

        return Plan(steps=plan_steps)

    def start_execution(self, plan: Plan, initial_data: Set[str], goal: Set[str]):
        """Initialize execution state and begin running the plan."""
        self.current_plan = plan
        self.current_step_index = 0

        # Set up execution state tracking
        self.execution_state = ExecutionState(
            current_propositions=initial_data.copy(),
            action_statuses={
                action.name: ActionStatus.PENDING
                for action in plan.get_all_actions()
            },
            goal=goal
        )

        print(f"Starting execution of {len(plan.steps)}-step plan.")
        print(f"Initial state: {initial_data}")
        print(f"Goal: {goal}")

    def execute_step(self) -> bool:
        """
        Execute the current step of the plan.

        Returns True if execution should continue, False if done or failed.
        """
        if self.current_plan is None or self.execution_state is None:
            return False

        if self.current_step_index >= len(self.current_plan.steps):
            # Check if we've achieved the goal
            if self.execution_state.goal.issubset(
                    self.execution_state.current_propositions
            ):
                print("Plan execution completed successfully!")
                return False
            else:
                print("Plan completed but goal not achieved. This shouldn't happen.")
                return False

        current_step = self.current_plan.steps[self.current_step_index]
        print(f"\nExecuting step {self.current_step_index + 1}: "
              f"{[a.name for a in current_step]}")

        # Check agent availability before executing
        for action in current_step:
            if action.agent_id not in self.registry.available_agents:
                print(f"Agent {action.agent_id} is unavailable!")
                return self._handle_agent_failure(action.agent_id)

        # Execute all actions in this step (they can run in parallel)
        for action in current_step:
            success = self._execute_action(action)
            if not success:
                # Agent became unavailable during execution
                return self._handle_agent_failure(action.agent_id)

        self.current_step_index += 1
        return True

    def _execute_action(self, action: Action) -> bool:
        """
        Execute a single action.

        In a real system, this would make an actual call to the agent.
        Here we simulate execution and update state accordingly.
        """
        # Simulate checking if agent is still available
        if action.agent_id not in self.registry.available_agents:
            self.execution_state.action_statuses[action.name] = ActionStatus.FAILED
            return False

        # Mark as executing
        self.execution_state.action_statuses[action.name] = ActionStatus.EXECUTING

        # Simulate the action (in reality, call the agent here)
        print(f"  Executing {action.name} on agent {action.agent_id}...")

        # Apply effects to our state model
        self.execution_state.current_propositions = action.apply(
            self.execution_state.current_propositions
        )

        # Mark as completed
        self.execution_state.action_statuses[action.name] = ActionStatus.COMPLETED
        print(f"  {action.name} completed. New state: "
              f"{self.execution_state.current_propositions}")

        return True

    def _handle_agent_failure(self, failed_agent_id: str) -> bool:
        """
        Handle an agent becoming unavailable.

        This is where the magic happens: we construct a new planning problem
        starting from our current state and try to find an alternative path
        to the goal using the remaining available agents.
        """
        print(f"\n{'='*60}")
        print(f"FAILURE DETECTED: Agent {failed_agent_id} is unavailable")
        print(f"{'='*60}")

        # Update registry
        self.registry.mark_unavailable(failed_agent_id)

        # What capabilities have we lost?
        alternatives = self.registry.find_alternative_agents(failed_agent_id)
        print(f"Lost capabilities and alternatives: {alternatives}")

        if not alternatives:
            print("No alternative agents available for lost capabilities.")
            print("Checking if remaining plan can still reach goal...")

        # Attempt repair by replanning from current state
        repair_plan = self._compute_repair_plan()

        if repair_plan is None:
            print("REPAIR FAILED: No alternative path to goal exists.")
            return False

        print(f"\nREPAIR SUCCESSFUL: Found alternative {len(repair_plan.steps)}-step plan")

        # Install the new plan and continue execution
        self.current_plan = repair_plan
        self.current_step_index = 0

        # Update action statuses for new plan
        for action in repair_plan.get_all_actions():
            if action.name not in self.execution_state.action_statuses:
                self.execution_state.action_statuses[action.name] = ActionStatus.PENDING

        return True

    def _compute_repair_plan(self) -> Optional[Plan]:
        """
        Compute a repair plan from the current execution state.

        The key insight: we're not starting from scratch. We're starting from
        wherever we managed to get to before the failure. All the data we've
        already produced is available, all the capabilities we've already
        achieved don't need to be re-achieved.
        """
        print(f"\nAttempting repair from state: "
              f"{self.execution_state.current_propositions}")
        print(f"Goal: {self.execution_state.goal}")

        # Get actions from available agents only
        available_actions = self.registry.get_available_actions()
        print(f"Available actions: {[a.name for a in available_actions]}")

        # Build a new planning graph from current state
        graph = PlanningGraph(
            initial_state=self.execution_state.current_propositions,
            goal=self.execution_state.goal,
            available_actions=available_actions
        )

        if not graph.build_graph():
            return None

        plan_steps = graph.extract_plan()
        if plan_steps is None:
            return None

        return Plan(steps=plan_steps)

    def run_to_completion(self) -> bool:
        """Execute the plan until completion or unrecoverable failure."""
        while self.execute_step():
            pass

        return self.execution_state.goal.issubset(
            self.execution_state.current_propositions
        )
# %%
def create_etablet_scenario() -> tuple[AgentRegistry, Set[str], Set[str]]:
    """
    Set up a scenario analogous to the paper's eTablet buying example,
    but with agents instead of web services.

    We'll have multiple agents that can provide similar capabilities,
    giving the system alternatives when one fails.
    """
    registry = AgentRegistry()

    # PearStoreAgent: can handle the complete purchase flow for Pear products
    pear_store_actions = [
        Action(
            name="pear_browse",
            agent_id="pear_store",
            preconditions=frozenset({"etablet_request"}),
            positive_effects=frozenset({"product_info", "pear_session"}),
            negative_effects=frozenset(),
            capability="product_selection"
        ),
        Action(
            name="pear_ship",
            agent_id="pear_store",
            preconditions=frozenset({"pear_session", "shipping_address"}),
            positive_effects=frozenset({"shipping_configured"}),
            negative_effects=frozenset(),
            capability="shipping_setup"
        ),
        Action(
            name="pear_pay",
            agent_id="pear_store",
            preconditions=frozenset({"pear_session", "credit_card"}),
            positive_effects=frozenset({"payment_complete"}),
            negative_effects=frozenset(),
            capability="payment"
        ),
        Action(
            name="pear_finalize",
            agent_id="pear_store",
            preconditions=frozenset({"pear_session", "shipping_configured",
                                     "payment_complete"}),
            positive_effects=frozenset({"tracking_number", "order_complete"}),
            negative_effects=frozenset({"pear_session"}),
            capability="order_finalization"
        ),
    ]
    registry.register_agent("pear_store", pear_store_actions)

    # GenericShopAgent: alternative shopping agent
    generic_shop_actions = [
        Action(
            name="generic_search",
            agent_id="generic_shop",
            preconditions=frozenset({"etablet_request"}),
            positive_effects=frozenset({"product_info", "generic_session"}),
            negative_effects=frozenset(),
            capability="product_selection"
        ),
        Action(
            name="generic_ship",
            agent_id="generic_shop",
            preconditions=frozenset({"generic_session", "shipping_address"}),
            positive_effects=frozenset({"shipping_configured", "shipping_cost"}),
            negative_effects=frozenset(),
            capability="shipping_setup"
        ),
        Action(
            name="generic_pay_card",
            agent_id="generic_shop",
            preconditions=frozenset({"generic_session", "credit_card"}),
            positive_effects=frozenset({"payment_complete"}),
            negative_effects=frozenset(),
            capability="payment"
        ),
        Action(
            name="generic_pay_paypal",
            agent_id="generic_shop",
            preconditions=frozenset({"generic_session", "paypal_token"}),
            positive_effects=frozenset({"payment_complete"}),
            negative_effects=frozenset(),
            capability="payment"
        ),
        Action(
            name="generic_finalize",
            agent_id="generic_shop",
            preconditions=frozenset({"generic_session", "shipping_configured",
                                     "payment_complete"}),
            positive_effects=frozenset({"tracking_number", "order_complete"}),
            negative_effects=frozenset({"generic_session"}),
            capability="order_finalization"
        ),
    ]
    registry.register_agent("generic_shop", generic_shop_actions)

    # PayPalAgent: payment provider
    paypal_actions = [
        Action(
            name="paypal_login",
            agent_id="paypal",
            preconditions=frozenset({"paypal_credentials"}),
            positive_effects=frozenset({"paypal_session"}),
            negative_effects=frozenset(),
            capability=None  # Auxiliary action, not a main capability
        ),
        Action(
            name="paypal_authorize",
            agent_id="paypal",
            preconditions=frozenset({"paypal_session", "shipping_cost"}),
            positive_effects=frozenset({"paypal_token"}),
            negative_effects=frozenset(),
            capability="payment"
        ),
    ]
    registry.register_agent("paypal", paypal_actions)

    # User's initial data
    initial_data = {
        "etablet_request",
        "shipping_address",
        "credit_card",
        "paypal_credentials"
    }

    # Goal: get the tracking number (order complete)
    goal = {"tracking_number"}

    return registry, initial_data, goal


def simulate_scenario():
    """
    Run through a scenario demonstrating self-healing.

    We'll start executing a plan, then simulate an agent failure
    partway through and watch the system recover.
    """
    print("="*70)
    print("SELF-HEALING AGENT COMPOSITION DEMONSTRATION")
    print("="*70)

    # Set up the scenario
    registry, initial_data, goal = create_etablet_scenario()

    print("\n--- Available Agents ---")
    for agent_id in registry.available_agents:
        actions = registry.agent_actions[agent_id]
        print(f"{agent_id}: {[a.name for a in actions]}")

    # Create the self-healing controller
    controller = SelfHealingController(registry)

    # Generate initial plan
    print("\n--- Generating Initial Plan ---")
    plan = controller.create_initial_plan(initial_data, goal)

    if plan is None:
        print("Could not generate initial plan!")
        return

    print(f"\nInitial plan has {len(plan.steps)} steps:")
    for i, step in enumerate(plan.steps):
        print(f"  Step {i+1}: {[a.name for a in step]}")

    # Start execution
    controller.start_execution(plan, initial_data, goal)

    # Execute first step normally
    print("\n--- Beginning Execution ---")
    controller.execute_step()

    # Now simulate a failure: the pear_store agent becomes unavailable
    print("\n" + "!"*70)
    print("SIMULATING FAILURE: pear_store agent goes offline!")
    print("!"*70)
    registry.mark_unavailable("pear_store")

    # Continue execution - the controller should detect the failure and repair
    success = controller.run_to_completion()

    print("\n" + "="*70)
    if success:
        print("SCENARIO COMPLETED SUCCESSFULLY")
        print("The system recovered from agent failure and achieved the goal.")
    else:
        print("SCENARIO FAILED")
        print("The system could not recover from the agent failure.")
    print("="*70)


if __name__ == "__main__":
    simulate_scenario()
# %%
