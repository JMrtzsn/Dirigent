// Package core provides the fundamental types for agent-to-agent communication.
package core

import (
	"time"

	"github.com/google/uuid"
)

// MessageKind discriminates message types for routing.
type MessageKind string

const (
	KindDelegateRequest  MessageKind = "delegate_request"
	KindDelegateResponse MessageKind = "delegate_response"
	KindFeedback         MessageKind = "feedback"
)

// DelegateRequest is sent when one agent asks another to perform work.
type DelegateRequest struct {
	ID            string
	Timestamp     time.Time
	Sender        string
	Receiver      string
	CorrelationID string
	Capability    string
	Task          string
	Context       map[string]any
	Constraints   map[string]any
}

// NewDelegateRequest creates a request with a generated ID and current timestamp.
func NewDelegateRequest(sender, receiver, capability, task string, ctx, constraints map[string]any) DelegateRequest {
	if ctx == nil {
		ctx = map[string]any{}
	}
	if constraints == nil {
		constraints = map[string]any{}
	}
	return DelegateRequest{
		ID:          uuid.New().String(),
		Timestamp:   time.Now(),
		Sender:      sender,
		Receiver:    receiver,
		Capability:  capability,
		Task:        task,
		Context:     ctx,
		Constraints: constraints,
	}
}

// DelegateResponse is the result returned from a delegated task.
type DelegateResponse struct {
	ID            string
	Timestamp     time.Time
	Sender        string
	Receiver      string
	CorrelationID string
	Success       bool
	Result        any
	Error         string
}

// NewDelegateResponse creates a response with a generated ID and current timestamp.
func NewDelegateResponse(sender, receiver, correlationID string, success bool, result any, errMsg string) DelegateResponse {
	return DelegateResponse{
		ID:            uuid.New().String(),
		Timestamp:     time.Now(),
		Sender:        sender,
		Receiver:      receiver,
		CorrelationID: correlationID,
		Success:       success,
		Result:        result,
		Error:         errMsg,
	}
}
