import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

export type WorkspaceChromeActions = {
  openSkills: () => void;
  requestDelete: () => void;
};

const RegisterContext = createContext<(actions: WorkspaceChromeActions) => void>(() => undefined);
const ActionsContext = createContext<WorkspaceChromeActions>({
  openSkills: () => undefined,
  requestDelete: () => undefined,
});
const BannerEndContext = createContext<ReactNode>(null);
const BannerEndRegisterContext = createContext<(node: ReactNode) => void>(() => undefined);

export function WorkspaceChromeProvider({ children }: { children: ReactNode }) {
  const actionsRef = useRef<WorkspaceChromeActions>({
    openSkills: () => undefined,
    requestDelete: () => undefined,
  });
  const register = useCallback((next: WorkspaceChromeActions) => {
    actionsRef.current = next;
  }, []);
  const actions = useMemo<WorkspaceChromeActions>(
    () => ({
      openSkills: () => actionsRef.current.openSkills(),
      requestDelete: () => actionsRef.current.requestDelete(),
    }),
    [],
  );
  const [bannerEnd, setBannerEnd] = useState<ReactNode>(null);
  return (
    <RegisterContext.Provider value={register}>
      <ActionsContext.Provider value={actions}>
        <BannerEndRegisterContext.Provider value={setBannerEnd}>
          <BannerEndContext.Provider value={bannerEnd}>{children}</BannerEndContext.Provider>
        </BannerEndRegisterContext.Provider>
      </ActionsContext.Provider>
    </RegisterContext.Provider>
  );
}

export function useWorkspaceChromeActions(): WorkspaceChromeActions {
  return useContext(ActionsContext);
}

export function useWorkspaceBannerEnd(): ReactNode {
  return useContext(BannerEndContext);
}

export function useRegisterWorkspaceChrome(actions: WorkspaceChromeActions) {
  const register = useContext(RegisterContext);
  const actionsRef = useRef(actions);
  actionsRef.current = actions;
  useEffect(() => {
    register({
      openSkills: () => actionsRef.current.openSkills(),
      requestDelete: () => actionsRef.current.requestDelete(),
    });
  }, [register]);
}

/** Put focusable controls in the workspace banner, right-aligned after the title. */
export function useRegisterBannerEnd(node: ReactNode) {
  const register = useContext(BannerEndRegisterContext);
  useEffect(() => {
    register(node);
    return () => register(null);
  }, [node, register]);
}
