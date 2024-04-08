  private static final Set<String> RESTRICTED_METHODS = ImmutableSet.<String> builder()
      .add("clone")
      .add("wait")
      .add("getClass")
      .add("notify")
      .add("hashCode")
      .add("notifyAll")
      .build();