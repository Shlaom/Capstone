""" common/models.py - Django Custom User Model 구현 """

from django.db import models
from django.contrib.auth.models import (BaseUserManager, AbstractBaseUser)


class UserManager(BaseUserManager):

    # username_field에 'userid'(사용자 아이디)를 사용할 것임
    def create_user(self, userid, username, password=None):
        if not userid:
            raise ValueError('Users must have an user id')
        if not username:
            raise ValueError('Users must have an user name')
        if not password:
            raise ValueError('must have user password')


        user = self.model(
            userid=userid,
            username=username,
        )

        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, userid, username, password):
        user = self.create_user(
            userid,
            password=password,
            username=username,
        )

        user.is_admin = True
        user.save(using=self._db)
        return user


class User(AbstractBaseUser):
    username = models.CharField(max_length=50, null=False, blank=False, verbose_name='사용자 이름')
    userid = models.CharField(max_length=50, primary_key=True, null=False, blank=False,
                              unique=True, verbose_name='사용자 아이디')

    # 아래 두 개의 필드는 Django의 User Model을 구성할 때 필수로 요구되는 항목
    is_active = models.BooleanField(default=True)
    is_admin = models.BooleanField(default=False)

    objects = UserManager()
    USERNAME_FIELD = 'userid'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.userid

    # True를 반환하여 권한이 있는 것을 알림
    # Object 반환 시 해당 Object로 사용 권한을 확인하는 절차가 필요함
    def has_perm(self, perm, obj=None):
        return True

    # True를 반환하여 주어진 App의 Model에 접근 가능하도록 함
    def has_module_perms(self, app_label):
        return True

    # True 반환 시 Django의 관리자 화면에 로그인 가능
    @property
    def is_staff(self):
        return self.is_admin
