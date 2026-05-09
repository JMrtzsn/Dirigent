package core_test

import (
	"testing"

	"github.com/johannesmauritzson/dirigent/core"
)

func TestRegistryRegisterAndResolveName(t *testing.T) {
	r := core.NewRegistry()
	ag := &fakeAgent{name: "dev", caps: []string{"implement"}}
	if err := r.Register(ag); err != nil {
		t.Fatal(err)
	}

	got, err := r.ResolveName("dev")
	if err != nil {
		t.Fatal(err)
	}
	if got.Info().Name != "dev" {
		t.Fatalf("expected dev, got %s", got.Info().Name)
	}
}

func TestRegistryDuplicateRegistration(t *testing.T) {
	r := core.NewRegistry()
	ag := &fakeAgent{name: "dev", caps: []string{"implement"}}
	_ = r.Register(ag)
	err := r.Register(ag)
	if err == nil {
		t.Fatal("expected error on duplicate registration")
	}
	regErr, ok := err.(*core.RegistryError)
	if !ok {
		t.Fatalf("expected RegistryError, got %T", err)
	}
	if regErr.Message == "" {
		t.Fatal("expected non-empty error message")
	}
}

func TestRegistryResolveCapability(t *testing.T) {
	r := core.NewRegistry()
	ag := &fakeAgent{name: "planner", caps: []string{"plan"}}
	_ = r.Register(ag)

	got, err := r.ResolveCapability("plan")
	if err != nil {
		t.Fatal(err)
	}
	if got.Info().Name != "planner" {
		t.Fatalf("expected planner, got %s", got.Info().Name)
	}
}

func TestRegistryUnknownCapability(t *testing.T) {
	r := core.NewRegistry()
	_, err := r.ResolveCapability("nonexistent")
	if err == nil {
		t.Fatal("expected error for unknown capability")
	}
	regErr, ok := err.(*core.RegistryError)
	if !ok {
		t.Fatalf("expected RegistryError, got %T", err)
	}
	if regErr.Message == "" {
		t.Fatal("expected non-empty error message")
	}
}

func TestRegistryUnknownName(t *testing.T) {
	r := core.NewRegistry()
	_, err := r.ResolveName("ghost")
	if err == nil {
		t.Fatal("expected error for unknown name")
	}
}

func TestCapabilityMatches(t *testing.T) {
	c := core.Capability{Name: "plan"}
	if !c.Matches("plan") {
		t.Fatal("expected match")
	}
	if c.Matches("implement") {
		t.Fatal("expected no match")
	}
}
