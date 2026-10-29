# Dockerfile Best Practices 

Below is a comprehensive Dockerfile best practices checklist, grouped by goal.

## Base images and reproducibility

- **Use an official, minimal base image:** Prefer maintained images and minimal variants appropriate for the application.
- **Pin base-image versions:** Use a specific language/runtime patch version rather than floating major or minor tags.
- **Pin base images by digest:** Use immutable SHA-256 image digests for fully reproducible builds.
- **Avoid mutable tags:** Do not use tags such as `latest` for base images, copied tooling, or dependencies.
- **Use compatible OS distributions across stages:** Builder and runtime stages should use compatible libc and package ecosystems when artifacts are copied between them.
- **Record image provenance:** Add OCI-standard labels for source repository, version, revision, build date, and license.

## Dependency management

- **Use lockfiles:** Install dependencies from a committed lockfile to ensure repeatable builds.
- **Pin application dependencies:** Define exact versions, ideally with hashes, for production dependencies.
- **Verify downloaded artifacts:** Validate SHA-256 checksums or signatures for source archives, binaries, and external tools.
- **Avoid unverified remote downloads:** Do not download and execute/build remote content without an integrity check.
- **Pin operating-system packages where appropriate:** Use controlled package versions when reproducibility requirements justify it.
- **Install only production dependencies:** Exclude development, testing, documentation, and debugging packages from runtime images.
- **Prefer prebuilt or trusted package sources:** Avoid compiling dependencies from source unless necessary.
- **Use offline installation in final stages:** Build or fetch artifacts earlier, then install from local artifacts without network access in the runtime stage.
- **Avoid unnecessary package-manager caches:** Disable or clean dependency caches when they do not improve the final image.

## Build stages and image size

- **Use multi-stage builds:** Keep compilers, headers, source code, and build tools out of the runtime image.
- **Copy only required artifacts between stages:** Transfer built binaries, wheels, static files, or application bundles—not entire build directories.
- **Use a dedicated builder stage:** Isolate compilers, package managers, and source-build dependencies.
- **Keep the final image minimal:** Include only the runtime, application, required shared libraries, and essential configuration.
- **Remove temporary files in the same layer:** Download, extract, build, and delete temporary artifacts in one `RUN` instruction.
- **Clean OS package metadata:** Remove package indexes after installation to reduce image size.
- **Use `--no-install-recommends` for APT:** Prevent optional packages from being installed unnecessarily.
- **Avoid installing duplicate packages:** Review package lists for overlapping tools and dependencies.
- **Avoid unnecessary diagnostic tools in production:** Remove shells, editors, `curl`, `wget`, and similar tools unless operationally required.
- **Do not copy the entire repository by default:** Copy only the files needed at each build step.

## Docker build-cache efficiency

- **Order instructions from least to most frequently changed:** Put base setup and dependency installation before application-source copies.
- **Copy manifests before source code:** Copy dependency manifests and lockfiles first so dependency layers are reused when source files change.
- **Use a `.dockerignore` file:** Exclude Git metadata, virtual environments, build output, tests where unnecessary, local secrets, and caches.
- **Keep related commands in a single layer:** Combine dependent package installation and cleanup commands to avoid retaining intermediate data.
- **Use BuildKit cache mounts where useful:** Cache package-manager downloads during builds without embedding them in image layers.
- **Avoid invalidating dependency layers unnecessarily:** Do not copy all source files before installing dependencies if only manifest files determine them.
- **Build only the required target stage:** Use named stages and build targets when development, testing, and production images differ.

## Security

- **Run as a non-root user:** Create a dedicated unprivileged account and switch to it before runtime.
- **Use a fixed non-root UID/GID:** Stable numeric IDs improve compatibility with orchestrators and mounted volumes.
- **Set ownership while copying files:** Use `COPY --chown` instead of copying as root and running recursive ownership changes later.
- **Minimize filesystem write access:** Ensure application directories are owned only where runtime writes are required.
- **Design for a read-only root filesystem:** Store temporary files, caches, uploads, and logs in explicitly writable mounted locations.
- **Do not bake secrets into images:** Never use `ARG`, `ENV`, `COPY`, or build layers for credentials, tokens, private keys, or environment files.
- **Use BuildKit secrets for build-time credentials:** Pass package-registry credentials or SSH keys through ephemeral secret mounts.
- **Do not expose confidential data in build logs:** Avoid commands that print secret values or sensitive configuration.
- **Minimize installed software:** Every package increases the vulnerability and maintenance surface.
- **Regularly rebuild images:** Rebuild periodically to receive patched base-image and OS-package dependencies.
- **Scan images in CI:** Scan for known vulnerabilities, leaked secrets, and misconfigurations before publication.
- **Generate an SBOM:** Produce a software bill of materials to support vulnerability tracking and compliance.
- **Sign published images:** Use image signing and verification in deployment pipelines where supported.
- **Use trusted registries and namespaces:** Restrict image sources and verify their origin.

## Runtime behavior

- **Use exec-form `CMD` or `ENTRYPOINT`:** This provides reliable argument handling and signal delivery.
- **Ensure graceful shutdown:** Make sure the application handles `SIGTERM` and exits cleanly.
- **Use one primary responsibility per container:** Keep a container focused on one service/process where possible.
- **Set only necessary environment variables:** Avoid redundant configuration and do not use environment variables for secrets unless injected securely at runtime.
- **Write logs to standard output/error:** Let Docker or the orchestration platform collect logs instead of writing unmanaged log files.
- **Use an explicit working directory:** Prevent accidental file placement and make runtime behavior predictable.
- **Expose ports only when needed:** Use `EXPOSE` to document the intended listening port; do not publish ports in the Dockerfile.
- **Define a meaningful health check:** Add `HEALTHCHECK` only when a reliable local health/readiness probe exists.
- **Avoid a misleading health check:** A process check alone may not indicate that dependencies, ports, or application initialization are healthy.
- **Set resource expectations outside the image:** Configure CPU, memory, restart policies, and replica counts through Compose, Kubernetes, or the platform configuration.

## Cross-platform and native-code builds

- **Build artifacts for the target platform:** Ensure compiled libraries and native Python/Node/other dependencies match the final image architecture.
- **Distinguish build and target platforms correctly:** Be careful when using Buildx variables such as `BUILDPLATFORM` and `TARGETPLATFORM`.
- **Avoid copying host-architecture binaries into target images:** This is especially important for compiled extensions, wheels, and C/C++ libraries.
- **Test each supported architecture:** Build and run validation tests for every published platform, such as `linux/amd64` and `linux/arm64`.
- **Use platform-aware package artifacts:** Prefer correctly tagged platform wheels or separate architecture-specific build stages.

## Shell and Dockerfile correctness

- **Fail fast in shell commands:** Use strict shell behavior, particularly `pipefail` when pipelines are used.
- **Use `&&` between dependent commands:** Stop the layer immediately if an earlier command fails.
- **Avoid relying on implicit shell behavior:** Use explicit paths, arguments, and expected file names.
- **Avoid broad wildcard installs where ambiguity is possible:** Explicitly identify the intended application artifact and dependency source.
- **Use clear named stages:** Names such as `builder`, `test`, and `runtime` improve maintainability.
- **Keep comments accurate:** Update comments when commands, versions, or package purposes change.
- **Avoid obsolete metadata conventions:** Prefer OCI labels over legacy labels such as `maintainer`.
- **Validate copied files exist:** Structure builds so missing configuration or artifacts fail clearly and early.

## Testing and delivery

- **Test the built image in CI:** Run the container and execute smoke tests against the final image, not only the source tree.
- **Test startup and shutdown behavior:** Confirm the service starts, reports healthy status when applicable, and handles termination correctly.
- **Test as the runtime user:** Ensure permissions work without root privileges.
- **Test without network access when intended:** Verify offline installation and runtime behavior if the image is expected not to need outbound access.
- **Test with a read-only filesystem:** Detect accidental writes to application or system directories.
- **Use immutable release tags:** Publish release images with versioned tags in addition to a convenience tag if one is maintained.
- **Promote the same image artifact:** Build once, then promote the exact digest through test, staging, and production environments.
- **Document build arguments and runtime configuration:** Clearly state supported platforms, required environment variables, ports, mounts, and health-check behavior.