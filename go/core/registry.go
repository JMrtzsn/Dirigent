package core

import "fmt"

// RegistryError is returned when capability or agent lookup fails.
type RegistryError struct {
	Message string
}

func (e *RegistryError) Error() string { return e.Message }

// Registry is an in-memory agent registry with capability-based lookup.
type Registry struct {
	agents map[string]Agent
}

// NewRegistry creates an empty registry.
func NewRegistry() *Registry {
	return &Registry{agents: map[string]Agent{}}
}

// Register adds an agent by its name. Returns an error if already registered.
func (r *Registry) Register(agent Agent) error {
	name := agent.Info().Name
	if _, exists := r.agents[name]; exists {
		return &RegistryError{Message: fmt.Sprintf("Agent '%s' already registered", name)}
	}
	r.agents[name] = agent
	return nil
}

// ResolveCapability finds an agent that provides the requested capability.
func (r *Registry) ResolveCapability(capability string) (Agent, error) {
	for _, ag := range r.agents {
		for _, cap := range ag.Info().Capabilities {
			if cap.Matches(capability) {
				return ag, nil
			}
		}
	}
	return nil, &RegistryError{Message: fmt.Sprintf("No agent provides capability '%s'", capability)}
}

// ResolveName finds an agent by name.
func (r *Registry) ResolveName(name string) (Agent, error) {
	ag, ok := r.agents[name]
	if !ok {
		return nil, &RegistryError{Message: fmt.Sprintf("Agent '%s' not found", name)}
	}
	return ag, nil
}

// ListAgents returns info for all registered agents.
func (r *Registry) ListAgents() []AgentInfo {
	infos := make([]AgentInfo, 0, len(r.agents))
	for _, ag := range r.agents {
		infos = append(infos, ag.Info())
	}
	return infos
}

// ListCapabilities returns all registered capability names.
func (r *Registry) ListCapabilities() []string {
	var caps []string
	for _, ag := range r.agents {
		for _, c := range ag.Info().Capabilities {
			caps = append(caps, c.Name)
		}
	}
	return caps
}
