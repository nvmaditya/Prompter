# Project Task Tracking

## Phase 0: Project Scaffolding
- [x] Create pyproject.toml, .gitignore, .env.example
- [x] Create prompter package (__init__, __main__, cli)
- [ ] Git init + initial commit
- [ ] pip install -e ".[dev]" and verify CLI

## Phase 1: Foundation Layer
- [ ] Create Pydantic data models (models/)
- [ ] Create PipelineState and config
- [ ] Create LLM client with retry logic (llm/)
- [ ] Create utilities (tokens, checkpoint, logging)
- [ ] Create unit tests and verify
- [ ] Git commit Phase 1

## Phase 2: Analyzer Agent
- [ ] Write analyzer system prompt
- [ ] Implement analyzer agent
- [ ] Wire into CLI
- [ ] Verify with test input

## Phase 3: Architect Agent
- [ ] Write architect system prompt
- [ ] Implement architect agent with technique registry
- [ ] Verify per-module prompt generation

## Phase 4: Communication Designer Agent
- [ ] Write communication designer system prompt
- [ ] Implement communication designer agent
- [ ] Verify inter-agent map generation

## Phase 5: Critic + Refiner Loop
- [ ] Write critic and refiner system prompts
- [ ] Implement critic agent with scoring
- [ ] Implement refiner agent
- [ ] Verify quality loop

## Phase 6: Packager + Output
- [ ] Implement JSON writer
- [ ] Implement Markdown writer
- [ ] Implement scaffold writer
- [ ] Verify all output formats

## Phase 7: LangGraph Integration
- [ ] Build StateGraph with all nodes/edges
- [ ] Full CLI implementation (generate + interactive)
- [ ] Checkpoint/resume support
- [ ] End-to-end verification

## Phase 8: Testing + Polish
- [ ] Integration tests for all agents
- [ ] Error scenario tests
- [ ] CLI polish and help text
- [ ] Final verification
