package core

import "context"

// Capability declares something an agent can do.
type Capability struct {
	Name        string
	Description string
}

// Matches checks if this capability satisfies a request (exact match).
func (c Capability) Matches(requested string) bool {
	return c.Name == requested
}

// AgentInfo holds static metadata about an agent.
type AgentInfo struct {
	Name         string
	Capabilities []Capability
}

// Agent is the interface all agents must satisfy.
type Agent interface {
	Info() AgentInfo
	Handle(ctx context.Context, req DelegateRequest) (DelegateResponse, error)
}
