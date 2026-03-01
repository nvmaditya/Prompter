# Lessons Learned

Patterns and corrections captured during development to prevent repeated mistakes.

---

## 1. pydantic-settings: populate_by_name required for keyword construction

**Problem**: `Settings(groq_api_key="test-key")` fails when using `validation_alias="GROQ_API_KEY"` because pydantic-settings only accepts the alias name by default.

**Fix**: Add `populate_by_name=True` to `SettingsConfigDict` so both the field name and alias work.

**Rule**: Always set `populate_by_name=True` when using `validation_alias` in BaseSettings if the settings will be constructed programmatically in tests or code.

## 2. setuptools build backend path

**Problem**: `setuptools.backends._legacy:_Backend` does not exist and causes `pip install -e .` to fail.

**Fix**: Use `setuptools.build_meta` as the build backend.

**Rule**: Standard setuptools build backend is `setuptools.build_meta`. Don't guess import paths.

## 3. State config dict excludes API key by design

**Problem**: `safe_dict()` strips `groq_api_key` from the settings for serialization into PipelineState. Agents cannot reconstruct full Settings from `state["config"]`.

**Fix**: Pass the real `Settings` object directly to agent functions as a parameter instead of reconstructing from state.

**Rule**: Agent functions should accept `settings: Settings | None = None` as a parameter. The state config is for serialization/checkpointing only, not for runtime credential access.
