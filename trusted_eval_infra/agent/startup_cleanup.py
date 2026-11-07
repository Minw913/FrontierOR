"""Stop the grader/gateway if agent setup fails before monitor startup."""
import functools
import logging


def install_startup_cleanup(manager_class):
    if manager_class.__dict__.get('_frontieror_startup_cleanup',False):return
    for name in ['start_all','resume_all']:
        original=getattr(manager_class,name)
        def wrap(function):
            @functools.wraps(function)
            def guarded(self,*args,**kwargs):
                try:return function(self,*args,**kwargs)
                except BaseException:
                    try:self.stop_all()
                    except Exception:
                        logging.getLogger(__name__).exception('Cleanup after agent startup failure failed')
                        self._stop_grader_daemon(timeout=2)
                    raise
            return guarded
        setattr(manager_class,name,wrap(original))
    manager_class._frontieror_startup_cleanup=True
